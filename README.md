# GANs on spheres
A pix2pix model on a sphere in Python to infer dark matter density from discrete galaxy positions

Note: paths need to be changed accordingly.

The code has been written with the help of the following sources: 
+ Martín Abadi et al. pix2pix: Image-to-image translation with a conditional gan: Tensorflow core,
December 2022. ([source](https://www.tensorflow.org/tutorials/generative/pix2pix))
+ Jason Brownlee. How to develop a pix2pix gan for image-to-image translation, January 2021. ([source](https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/))
+ Nathanaël Perraudin et al. DeepSphere: Efficient spherical convolutional neural network with HEALPix sampling for cosmological applications, October 2018. ([paper](https://arxiv.org/abs/1810.12186))
+ Matthew A. Petroff et al. Fullsky cosmic microwave background foreground cleaning using machine learning, April 2020. ([paper](https://arxiv.org/abs/2004.11507))
+ pawangfg. Image-to-image translation using pix2pix, June 2022. ([source](https://www.geeksforgeeks.org/image-to-image-translation-using-pix2pix/))
+ Francisco Villaescusa-Navarro et al. The CAMELS project: Cosmology and astrophysics with machine-learning simulations, July 2021. ([paper](https://doi.org/10.3847%2F1538-4357%2Fabf7ba))


The deepsphere folder is from the [deepsphere GiHub repository](https://github.com/deepsphere/deepsphere-cosmo-tf2) (installation of packages has been performed according to the instructions in the deepsphere repository). Furthermore, the sccn folder is a supplement to the [paper](https://arxiv.org/abs/2004.11507) by Petroff et al. and can be found [here](https://zenodo.org/record/3764069#.ZABja4DMIvk).

The data was unfortunately too large for this repository. However, the snap files (snap_033.hdf5) can be found [here](https://users.flatironinstitute.org/~camels/Sims/IllustrisTNG/) in different CV repositories. Regarding the hlist files (hlists/hlist_1.00000.list), one can download them from [here](https://users.flatironinstitute.org/~camels/Rockstar/IllustrisTNG/) from the different CV repositories, fully analogously to the snap files.
