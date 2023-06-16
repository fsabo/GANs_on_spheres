import numpy as np
import healpy as hp
import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, initializers, losses
#from deepsphere import healpy_layers as hp_layer
import tensorflow as tf
import scnn.layers
import scnn.dropout
#%%
physical_devices=tf.config.list_physical_devices("GPU")
print("Num GPUs:", len(physical_devices))
#%%
print("load in the data")
train_set=np.load('../3_channels/train_set.npy').astype(np.float32)[...,[0,1]]
test_set=np.load('../3_channels/test_set.npy').astype(np.float32)[...,[0,1]]
print(train_set.shape)
print(test_set.shape)
#%%
nside = 64
healpix_img = np.zeros(hp.nside2npix(nside))
npix = len(healpix_img)
#%%
LAMBDA = 100
loss_object = losses.BinaryCrossentropy()
def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)
  return total_gen_loss, l1_loss, gan_loss
#%%
def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss
#%%
def define_discriminator(image_shape,output_image):
    # weight initialization
    # source image input
    in_src_image = layers.Input(shape=image_shape)
    # target image input
    in_target_image = layers.Input(shape=output_image)
    # concatenate images channel-wise
    merged = layers.Concatenate()([in_src_image, in_target_image])
    
    downscale = []
    # Construct inputs, one per input frequency
    # Construct coarsening / pooling layers, separate for each frequency
    for i in range(num_downscales):
        #downscale.append([])
        if i != 0:
            # Pool all but first layer
            downscale.append(scnn.layers.GraphPool(4)(downscale[-1]))
        else:
            downscale.append(tf.keras.layers.Reshape((hp.nside2npix(nside), 2))(merged))
        downscale.append(scnn.layers.SphereConvolution(
                filter_size=nsides_up[i], poly_k=poly_k[i], nside=nsides[i]
            )(downscale[-1]))
        downscale.append(layers.BatchNormalization()(downscale[-1]))
        downscale.append(layers.LeakyReLU(alpha=0.2)(downscale[-1]))
        downscale.append(scnn.layers.SphereConvolution(
                filter_size=nsides_up[i], poly_k=poly_k[i], nside=nsides[i]
            )(downscale[-1]))
        downscale.append(layers.BatchNormalization()(downscale[-1]))
        downscale.append(layers.LeakyReLU(alpha=0.2)(downscale[-1]))

    d=scnn.layers.SphereConvolution(
            filter_size=1, poly_k=poly_k[-1], nside=2
        )(downscale[-1])
    d=layers.Activation('sigmoid')(d)
    d=tf.math.reduce_mean(tf.math.reduce_mean(d,axis=2),axis=1)
    print("d shape:{}".format(d.shape))
            
    # define model
    model = models.Model([in_src_image, in_target_image], d)
    return model
#%%
# define the standalone generator model
def define_generator(image_shape):
    # weight initialization
    #init = initializers.RandomNormal(stddev=0.02)
    # image input
    in_image = layers.Input(shape=image_shape)
    downscale = []
    # Construct inputs, one per input frequency
    # Construct coarsening / pooling layers, separate for each frequency
    for i in range(num_downscales):
        #downscale.append([])
        if i != 0:
            # Pool all but first layer
            downscale.append(scnn.layers.GraphPool(4)(downscale[-1]))
        else:
            downscale.append(tf.keras.layers.Reshape((hp.nside2npix(nside), 1))(in_image))
        downscale.append(scnn.layers.SphereConvolution(
                filter_size=nsides_up[i], poly_k=poly_k[i], nside=nsides[i]
            )(downscale[-1]))
        downscale.append(layers.LeakyReLU(alpha=0.2)(downscale[-1]))
        downscale.append(scnn.layers.SphereConvolution(
                filter_size=nsides_up[i], poly_k=poly_k[i], nside=nsides[i]
            )(downscale[-1]))
        downscale.append(layers.LeakyReLU(alpha=0.2)(downscale[-1]))
    # Construct scaling / biasing for adding to upscaled layers, separate for each frequency
    for i in range(len(downscale)):
        downscale[i]=scnn.layers.LinearCombination()(downscale[i])
    # Add separate fully-coarsened frequency layers into a single layer for upscaling
    upscale = tf.keras.layers.add([downscale[-1],downscale[-1]])/2
    #upscale = layers.LeakyReLU(alpha=0.2)(upscale)
    upscale = tf.keras.layers.UpSampling1D(4)(upscale)
    upscale = tf.keras.layers.Activation("relu")(upscale)
    # Construct upscaling / skip connection layers
    for i in range(1,num_downscales):
        upscale = scnn.layers.SphereConvolution(
                filter_size=nsides_up[num_downscales + i],
                poly_k=poly_k[num_downscales + i],
                nside=nsides[num_downscales + i],
            )(upscale)
        upscale = layers.Concatenate()([upscale, downscale[-1 - 5*i]])  # Skip connection
        if i < num_downscales - 1:
            # Upsample all but last layer
            upscale = tf.keras.layers.UpSampling1D(4)(upscale)
        upscale = tf.keras.layers.Activation("relu")(upscale)

    # Final convolution layer
    upscale = scnn.layers.SphereConvolution(
            filter_size=nsides_up[-1], poly_k=poly_k[-1], nside=nsides[-1]
        )(upscale)
    upscale = layers.Activation('sigmoid')(upscale)

    return tf.keras.models.Model(inputs=in_image, outputs=upscale)
#%%
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    # define the source image
    in_src = layers.Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = models.Model(in_src, [dis_out, gen_out])
    return model
#%%
print("optimizers")
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
@tf.function
#@njit(parallel=True)
def train_step(input_image, target):
    #print("0")
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        #print("1")
        gen_output = g_model(input_image,training=True)
        #print("2")
        disc_real_output = d_model([input_image, target],training=True)
        #print("3")
        disc_generated_output = d_model([input_image, gen_output],training=True)
        #print("4")
        gen_total_loss,l1_l,gan_l = generator_loss(disc_generated_output, gen_output, target)
        #print("5")
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    #print("6")
    generator_gradients = gen_tape.gradient(gen_total_loss,g_model.trainable_variables)
    #print("7")
    discriminator_gradients = disc_tape.gradient(disc_loss,d_model.trainable_variables)
    #print("8")
    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            g_model.trainable_variables))
    #print("9")
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                d_model.trainable_variables))
    #print("10")
    #return None
    return gen_total_loss, disc_loss
#%%
# train pix2pix model
#@tf.function
def train(d_model, g_model, gan_model, dataset,n_epochs=10, n_batch=1):
    # unpack dataset
    trainA, trainB = dataset[...,0][:,:,None],dataset[...,1:]
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    disc_loss=[]
    gen_loss=[]
    for i in tqdm.tqdm(range(n_steps)):
        #print("step {}".format(i))
        ix = np.random.randint(0, trainA.shape[0], n_batch)
        # retrieve selected ima
        X1, X2 = trainA[ix], trainB[ix]
        g1,g2 = train_step(X1, X2)
        gen_loss.append(g1)
        disc_loss.append(g2)
        if i%(5*bat_per_epo)==0 and i!=0:
            g_model.save("model_1_epochs_{}.h5".format(i/bat_per_epo))
            np.save("g_loss_1.npy",gen_loss)
            np.save("d_loss_1.npy",disc_loss)
    return disc_loss,gen_loss
#%%
# define input shape based on the loaded dataset
image_shape = (npix,1)
output_image = (npix,1)
nsides = []
nsides_up = []
i = nside
j=16
while i >= 2:
    nsides.append(i)
    nsides_up.append(j)
    i = i // 2
    j=j*2
#nsides+=[1]
#nsides_up+=[j]
nsides += reversed(nsides)
nsides_up += reversed(nsides_up)
nsides_up+=[1]

print(nsides)
print(nsides_up)

FILTER_MAPS = 6
POLY_ORDER = 5
LENGTH_SCALE = 1e-4

# Construct filter maps and Chebyshev polynomial order lists
filters = [FILTER_MAPS] * len(nsides) + [1]  # Result + uncertainty outputs
poly_k = [POLY_ORDER] * len(nsides)

n = train_set.shape[1]
wd = LENGTH_SCALE ** 2.0 / n
dd = 2.0 / n

num_downscales = len(nsides) // 2
input_channels=1
print(input_channels)
# define the model
print("discriminator")
d_model = define_discriminator(image_shape,output_image)
print("generator")
g_model = define_generator(image_shape)
print("Discriminator:")
print(d_model.summary())
print("Generator:")
print(g_model.summary())
gan_model = define_gan(g_model, d_model, image_shape)
# train model
print("training the model...")
n_epochs=400
print(train_set.shape)
print(test_set.shape)
d_loss,g_loss=train(d_model, g_model, gan_model, train_set,n_epochs)
#%%
# generate image from source
g_model.save("model_1_final.h5")
np.save("d_loss_1_final.npy",d_loss)
np.save("g_loss_1_final.npy",g_loss)
