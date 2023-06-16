import numpy as np
import healpy as hp
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import scnn.layers
import scnn.dropout
train_set=np.load('3_channels/train_set.npy').astype(np.float32)[...,[0,1]]
test_set=np.load('3_channels/test_set.npy').astype(np.float32)[...,[0,1]]
model_1= tf.keras.models.load_model('density/model_2_final.h5', custom_objects={'SphereConvolution': scnn.layers.SphereConvolution, 
                                                                       'GraphPool':scnn.layers.GraphPool,
                                                                       'LinearCombination':scnn.layers.LinearCombination})
titles=['Source','Generated','Real',"Angular power spectrum","Histogramm"]
ll = np.arange(192)
def plot_images(data,model,name):
    n=5
    gen_image = model.predict(data[...,0])
    sets=np.concatenate((data[...,0][:,:,None],gen_image,data[...,1:]),axis=2)
    fig,axs=plt.subplots(n,5,figsize=(20,20))
    fig.tight_layout()
    for i in tqdm.tqdm(range(n)):
        idx=np.random.randint(low=0,high=data.shape[0]-1)
        for j in range(5):
            if j<3:
                plt.axes(axs[i%n,j])
                hp.mollview(sets[idx,:,j].flatten(),cmap='turbo',hold=True,nest=True,title=titles[j],min=0,max=1)
                hp.graticule()
            else:
                if j==3:
                    axs[i%n,j].plot(ll,hp.anafast(sets[idx,:,1]),color="red",label="generated")
                    axs[i%n,j].plot(ll,hp.anafast(sets[idx,:,2]),color="blue",label="real")
                    axs[i%n,j].plot(ll,hp.anafast(sets[idx,:,1])/hp.anafast(sets[idx,:,2]),color="black",label="generated vs. real")
                    axs[i%n,j].legend()
                    axs[i%n,j].grid()
                    axs[i%n,j].set_yscale("log")
                    axs[i%n,j].set_title(titles[j])
                else:
                    plt.axes(axs[i%n,j])
                    sns.kdeplot(sets[idx,:,1],color="red",label="generated",fill=True)
                    sns.kdeplot(sets[idx,:,2],color="blue",label="real",fill=True)
                    plt.legend()
                    plt.grid()
                    plt.xlim([0,1])
                    plt.title(titles[j])
    plt.savefig("aps_analysis_{}_model_{}.pdf".format(name[0],name[1]))
plot_images(train_set,model_1,["training_set","1"])
plot_images(test_set,model_1,["test_set","1"])         
        
    
