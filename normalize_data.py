import numpy as np
train_set_1=np.load('data/train_set_64.npy')
test_set_1=np.load('data/test_set_64.npy')
train_set_2=np.load('data/train_set_64_metal.npy')
test_set_2=np.load('data/test_set_64_metal.npy')
train_set_3=np.load('data/train_set_64_temp.npy')
test_set_3=np.load('data/test_set_64_temp.npy')
def norm_data(train,test):
    data=np.concatenate((train,test),axis=1)
    galaxy_array=np.interp(data[0], (np.array(data[0]).min(), np.array(data[0]).max()), (0, +1))
    gas_array = np.log10(data[1])
    gas_array=np.interp(gas_array, (gas_array.min(), gas_array.max()), (0, +1))
    return np.array([galaxy_array[:-250],gas_array[:-250]]),np.array([galaxy_array[-250:],gas_array[-250:]])
def merge_sets(set1,set2,set3):
    return np.concatenate((set1[0],set1[1],set2[1],set3[1]),axis=2)
train_set_1,test_set_1=norm_data(train_set_1,test_set_1)
train_set_2,test_set_2=norm_data(train_set_2,test_set_2)
train_set_3,test_set_3=norm_data(train_set_3,test_set_3)
data=merge_sets(train_set_1,train_set_2,train_set_3)
print(data.shape)
np.save('train_set.npy', data) # save
data=merge_sets(test_set_1,test_set_2,test_set_3)
print(data.shape)
np.save('test_set.npy', data) # save



