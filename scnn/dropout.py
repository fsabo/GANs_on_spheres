"""
Concrete dropout module.

Based on ConcreteDropout (https://github.com/yaringal/ConcreteDropout),
`spatial-concrete-dropout-keras.ipynb` file, commit 7276547054bee48b6ccd9daf75600815a1de53b4.

Copyright (c) 2017-2018 Yarin Gal
Copyright (c) 2019-2020 Matthew Petroff

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec, Wrapper
import tensorflow as tf


class SpatialConcreteDropout(Wrapper):
    """This wrapper allows to learn the dropout probability for any given Conv1D input layer.
    ```python
        model = Sequential()
        model.add(SpatialConcreteDropout(Conv1D(64, 3),
                                         input_shape=(299, 3)))
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss and used only for the Euclidean loss.
    """

    def __init__(
        self,
        layer,
        weight_regularizer=1e-6,
        dropout_regularizer=1e-5,
        init_min=0.1,
        init_max=0.1,
        is_mc_dropout=True,
        data_format=None,
        **kwargs
    ):
        assert "kernel_regularizer" not in kwargs
        super().__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.init_min = np.log(init_min) - np.log(1.0 - init_min)
        self.init_max = np.log(init_max) - np.log(1.0 - init_max)
        self.data_format = "channels_last" if data_format is None else "channels_first"

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super().build()  # This is very weird... We must call super before we add new losses

        # Initialize p
        self.p_logit = self.layer.add_weight(
            name="p_logit",
            shape=(1,),
            initializer=tf.initializers.RandomUniform(self.init_min, self.init_max),
            trainable=True,
        )
        self.p = K.sigmoid(self.p_logit[0])

        # Initialize regularizer / prior KL term
        assert len(input_shape) == 3, "This wrapper only supports Conv1D layers"
        if self.data_format == "channels_first":
            input_dim = input_shape[1]  # We drop only channels
        else:
            input_dim = input_shape[2]

        weight = self.layer.kernel
        kernel_regularizer = (
            self.weight_regularizer * K.sum(K.square(weight)) / (1.0 - self.p)
        )
        dropout_regularizer = self.p * K.log(self.p)
        dropout_regularizer += (1.0 - self.p) * K.log(1.0 - self.p)
        dropout_regularizer *= self.dropout_regularizer * int(input_dim)
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def spatial_concrete_dropout(self, x):
        """
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        """
        eps = K.cast_to_floatx(K.epsilon())
        temp = 2.0 / 3.0  # Concrete distribution temperature

        unif_noise = tf.random.uniform(shape=tf.shape(input=x))

        # If we use self.p, tf.saved_model.save fails...
        p = K.sigmoid(self.p_logit[0])
        drop_prob = (
            K.log(p + eps)
            - K.log(1.0 - p + eps)
            + K.log(unif_noise + eps)
            - K.log(1.0 - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1.0 - drop_prob

        retain_prob = 1.0 - p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.spatial_concrete_dropout(inputs))
        else:

            def relaxed_dropped_inputs():
                return self.layer.call(self.spatial_concrete_dropout(inputs))

            return K.in_train_phase(
                relaxed_dropped_inputs, self.layer.call(inputs), training=training
            )
