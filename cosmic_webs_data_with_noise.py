import numpy as np
import h5py
import healpy as hp
import tqdm
import pandas as pd
from numba import njit
from scipy.interpolate import NearestNDInterpolator
#%%
def file_to_pandas(name):
    data=[]
    with open(name,'r') as f:
        my_list = f.read()
        for line in my_list.splitlines():
            data.append(line.split())
    return pd.DataFrame(np.array(data[64:]).astype(float),columns=data[0])
#%%
PATH="../../../../mnt/data/filipsabo"
n_files=16
BoxSizes=[]
pos_gs=[]
rho_gs=[]
interp_rhos=[]
galaxies_datasets=[]
print("importing files...")
for i in tqdm.tqdm(range(n_files)):
    with h5py.File(PATH+'/snap/snap_033_{}.hdf5'.format(i)) as f:
        BoxSize  = f['Header'].attrs[u'BoxSize']/1e3  # Mpc/h
        pos_g    = f['PartType0/Coordinates'][:]/1e3  # Mpc/h
        pos_g    = pos_g.astype(np.float32)           # positions as float32
        pos_g    = np.fmod(pos_g,BoxSize)    # periodically wrap
        rho_g    = f['PartType0/Density'][:]   # 1e10 h-1 Msun / ( h-1 ckpc)^3
    BoxSizes.append(BoxSize)
    pos_gs.append(pos_g)
    rho_gs.append(rho_g)
    interp_rhos.append(NearestNDInterpolator(pos_gs[-1], rho_gs[-1], tree_options={'boxsize' : BoxSizes[-1]}))
    galaxies_datasets.append(file_to_pandas(PATH+'/hlist/hlist_{}_1.00000.list'.format(i)))
#%%
nside = 64
npix = hp.nside2npix(nside)
indices=np.arange(hp.nside2npix(nside))
#%%
theta, phi = hp.pix2ang(nside, np.arange(npix), nest=True)
theta_phi = np.stack((theta, phi), -1)
pi_minus_theta = np.pi - theta
pi_minus_theta_phi = np.stack((pi_minus_theta, phi), -1)
two_pi_minus_phi = (2*np.pi - phi)%(2*np.pi)
two_pi_minus_theta_phi = np.stack((theta, two_pi_minus_phi), -1)
#%%
@njit
def build_inds(minus_array,th_ph):
    inds_i = []
    inds_j = []
    for i, p1 in enumerate(minus_array):
        for j, p2 in enumerate(th_ph):
            if np.sum(np.abs(p1 - p2)) < 1e-10:
                inds_i.append(i)
                inds_j.append(j)
    return inds_i, inds_j
#%%
print("theta")
inds_i_1, inds_j_1 = build_inds(pi_minus_theta_phi,theta_phi)
print("phi")
inds_i_2, inds_j_2 = build_inds(two_pi_minus_theta_phi,theta_phi)
sigma=5*10**(-2)
#%%
def gen_query_points( x0, r, npix ):
    theta,phi = hp.pix2ang(nside,range(npix))
    pos_healpy = np.zeros((npix,3))
    pos_healpy[:,0] = x0[0] + r * np.sin(theta) * np.cos(phi)
    pos_healpy[:,1] = x0[1] + r * np.sin(theta) * np.sin(phi)
    pos_healpy[:,2] = x0[2] + r * np.cos(theta)
    return pos_healpy
#%%
def gen_avg_maps(x0, rmin, rmax, nsteps, interp):
    rr = np.linspace(rmin,rmax,nsteps)
    rho_img = np.zeros(hp.nside2npix(nside))
    for r in rr:
        pos_healpy = gen_query_points( x0=x0, r=r, npix=npix)
        rho_slice  = interp( pos_healpy )
        rho_img   += rho_slice
    return rho_img/nsteps
#%%
def get_spherical(galaxy,x0,size):
    gal_pos=galaxy[["x(17)","y(18)","z(19)"]].to_numpy()
    
    rgal_pos = gal_pos - x0
    rgal_pos[rgal_pos<-size/2] += size
    rgal_pos[rgal_pos>size/2]  -= size
    
    r_gal = np.sqrt((rgal_pos**2).sum(axis=1))
    n1 = np.random.normal(0,sigma,r_gal.shape[0])
    n2 = np.random.normal(0,sigma,r_gal.shape[0])
    
    phi_gal = np.arctan2(rgal_pos[:,1],rgal_pos[:,0])
    theta_gal = np.arccos(rgal_pos[:,2]/r_gal)
    
    phi_gal += n1
    theta_gal += n2 * np.sin(theta_gal)
    
    iflip = theta_gal < 0
    theta_gal[iflip] = -theta_gal[iflip]
    #phi_gal[iflip] += np.pi
    
    iflip2 = theta_gal > np.pi
    theta_gal[iflip2] = 2*np.pi - theta_gal[iflip2]
    #phi_gal[iflip2] += np.pi

    return np.c_[r_gal,theta_gal,phi_gal]
#%%
def get_counts(theta,phi):
    # convert to HEALPix indices
    indices = hp.ang2pix(nside,theta,phi)

    idx, counts = np.unique(indices, return_counts=True)

    # fill the fullsky map
    hpx_map = np.zeros(npix, dtype=int)
    hpx_map[idx] = counts
    return hpx_map
#%%
def perform_cuts(x0,galaxy,gas,sizes,augmentation=True):
    galaxy_array=[]
    gas_array=[]
    for i in tqdm.tqdm(range(len(galaxy))):
        for j in range(5):
            galaxy_positions_spherical=get_spherical(galaxy[i],x0[j,:],sizes[i])
            
            ii = (galaxy_positions_spherical[:,0]>=r_min)&(galaxy_positions_spherical[:,0]<=r_max)
            galaxy_array.append(hp.pixelfunc.reorder(get_counts(galaxy_positions_spherical[ii,1],galaxy_positions_spherical[ii,2]),r2n=True))
            
            gas_array.append(hp.pixelfunc.reorder(gen_avg_maps(x0[j], r_min, r_max, 20, gas[i]),r2n=True))
    galaxy_array=np.interp(galaxy_array, (np.array(galaxy_array).min(), np.array(galaxy_array).max()), (0, +1))
    gas_array = np.log10(gas_array)
    gas_array=np.interp(gas_array, (gas_array.min(), gas_array.max()), (0, +1))
    if augmentation:
        for i in tqdm.tqdm(range(galaxy_array.shape[0])):
            img = galaxy_array[i,:]
            galaxy_array = np.r_[galaxy_array,img[inds_j_1][None,:]]
            galaxy_array = np.r_[galaxy_array,img[inds_j_2][None,:]]
            galaxy_array = np.r_[galaxy_array,img[inds_j_1][inds_j_2][None,:]]
                   
            img_1 = gas_array[i,:]
            gas_array = np.r_[gas_array,img_1[inds_j_1][None,:]]
            gas_array = np.r_[gas_array,img_1[inds_j_2][None,:]]
            gas_array = np.r_[gas_array,img_1[inds_j_1][inds_j_2][None,:]]
            
    return galaxy_array[:,:,None].astype(np.float32),gas_array[:,:,None].astype(np.float32)
#%%
x_grid=np.arange(0,25,5)
x0s=np.array(np.meshgrid(x_grid,x_grid,x_grid)).T.reshape(-1,3)
r_min=8
r_max=10

print("performing cuts for the test set...")
N=100
test_set=[]
for i in tqdm.tqdm(range(N)):
    test_set.append(perform_cuts(x0s,galaxies_datasets[-2:],interp_rhos[-2:],BoxSizes[-2:],False))
np.save('test_set_64_noise_multiple.npy', test_set) # save
