import numpy as np
import healpy as hp
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import scnn.layers
import scnn.dropout
test_set=np.load('3_channels/test_set.npy').astype(np.float32)[...,[0,1]]
test_noise=np.load('test_set_64_noise_multiple.npy')
print(test_noise.shape)
print(test_set.shape)
model_1= tf.keras.models.load_model('density/model_1_final.h5', custom_objects={'SphereConvolution': scnn.layers.SphereConvolution, 
                                                                       'GraphPool':scnn.layers.GraphPool,
                                                                       'LinearCombination':scnn.layers.LinearCombination})
titles=['Source','Real','Avg. of generated with noise','Std. of generated with noise',"Relative variance"]
def plot_images(data,noise,model,name):
    fig,axs=plt.subplots(5,5,figsize=(15,10))
    fig.tight_layout()
    for i in tqdm.tqdm(range(len(idxs))):
        image=[]
        image_generated=[]
        for ii in tqdm.tqdm(range(noise.shape[0])):
            image_generated.append(model.predict(noise[ii,0,idxs_noise[i]][None,:]).flatten())
        image_generated=np.array(image_generated)
        print(image_generated.shape)
        for j in range(5):
            if j==0:
                image.append(data[idxs[i],:,0])
            else:
                if j==1:
                    image.append(data[idxs[i],:,1])
                else:
                    if j==2:
                        image.append(np.mean(image_generated,axis=0))
                    else:
                        if j==3:
                            image.append(np.std(image_generated,axis=0))
                        else:
                            image.append(np.std(image_generated,axis=0)/np.mean(image_generated,axis=0))
            plt.axes(axs[i%5,j])
            hp.mollview(image[-1].flatten(),cmap='turbo',hold=True,nest=True,title=titles[j],min=0,max=upper[j])
            hp.graticule()
    plt.savefig("noise_aps_analysis_{}_model_{}.pdf".format(name[0],name[1]))
idxs=[0,3,126,128,129]
idxs_noise=[0,3,6,8,9]
upper=[1,1,1,0.15,0.6]

plot_images(test_set,test_noise,model_1,["test_set","1"])
