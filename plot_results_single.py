import numpy as np
import healpy as hp
import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import scnn.layers
import scnn.dropout
train_set=np.load('3_channels/train_set.npy').astype(np.float32)[...,[0,1]]
test_set=np.load('3_channels/test_set.npy').astype(np.float32)[...,[0,1]]
print(test_set.shape)
"""model_1= tf.keras.models.load_model('density/model_2_epochs_5.0.h5', custom_objects={'SphereConvolution': scnn.layers.SphereConvolution, 
                                                                       'GraphPool':scnn.layers.GraphPool,
                                                                       'LinearCombination':scnn.layers.LinearCombination})"""
model_2= tf.keras.models.load_model('density/model_2_final.h5', custom_objects={'SphereConvolution': scnn.layers.SphereConvolution, 
                                                                       'GraphPool':scnn.layers.GraphPool,
                                                                       'LinearCombination':scnn.layers.LinearCombination})
def plot_images(data,model,addition):
    gen_image = model.predict(data[...,0])
    sets=np.concatenate((data[...,0][:,:,None],gen_image,data[...,1:]),axis=2)
    titles=['Source','Generated - ','Real - ']
    fig,axs=plt.subplots(5,3,figsize=(15,16))
    fig.tight_layout()
    for i in tqdm.tqdm(range(5)):
        idx=np.random.randint(low=0,high=gen_image.shape[0]-1)
        for j in range(3):
            plt.axes(axs[i%5,j])
            hp.mollview(sets[idx,:,j],cmap='turbo',hold=True,nest=True,title=titles[j],min=0,max=1)
            hp.graticule()
    plt.savefig("points_to_field_{}.pdf".format(addition))
#plot_images(train_set,model_1,"train_1")
#plot_images(train_set,model_2,"train_2")
#plot_images(test_set,model_1,"test_1")
plot_images(test_set,model_2,"test_2")
