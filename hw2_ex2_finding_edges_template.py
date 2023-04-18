""" 2 Finding edges """

import numpy as np
from skimage import color, io
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
import pdb
import hw2_ex1_linear_filtering_template as ex1

# load image
img = io.imread('bird.jpg')
img = color.rgb2gray(img)


### copy functions myconv2, gauss1d, gauss2d and gconv from exercise 1 ###


# 2.1
# Gradients
# define a derivative operator
dx = np.array([-1,0,1])### your code should go here ###
dy = np.transpose(dx)### your code should go here ###

# convolve derivative operator with a 1d gaussian filter with sigma = 1
# You should end up with 2 1d edge filters,  one identifying edges in the x direction, and
# the other in the y direction
sigma = 1
### your code should go here ###
filterlenght=3
gdx=ex1.myconv2(dx,np.asarray(ex1.gauss1d(sigma,filterlenght)))
inbuiltresult=np.convolve(dx,ex1.gaussian_filter1d(np.ones(filterlenght),sigma))
if np.isclose(gdx, inbuiltresult).all:
    print("my gdx works")
else:
    print("(my gdx doesn't work")

gdy=ex1.myconv2(dy,np.asarray(ex1.gauss1d(sigma,filterlenght)))
inbuiltresult=np.convolve(dy,ex1.gaussian_filter1d(np.ones(filterlenght),sigma))
if np.isclose(gdy, inbuiltresult).all:
    print("my gdy works")
else:
    print("(my gdy doesn't work")

# gdx = ### your code should go here ###
# gdy = ### your code should go here ###


# 2.2
# Gradient Edge Magnitude Map
def create_edge_magn_image(image, gdx, gdy):
    # this function created an eddge magnitude map of an image
    # for every pixel in the image, it assigns the magnitude of gradients
    # INPUTS
    # @image  : a 2D image
    # @gdx     : gradient along x axis
    # @gdy     : gradient along y axis
    # OUTPUTS
    # @ grad_mag_image  : 2d image same size as image, with the magnitude of gradients in every pixel
    # @grad_dir_image   : 2d image same size as image, with the direction of gradients in every pixel

    ### your code should go here ###
    Ix = ex1.myconv2(image, gdx)
    Iy = ex1.myconv2(image, gdy)
    # cutting step:
    Ix = Ix[:image.shape[0], :image.shape[1]]
    Iy = Iy[:image.shape[0], :image.shape[1]]

    grad_mag_image = np.hypot(Ix, Iy)  # directly calculates the magnitude for each pixel
    grad_dir_image = np.arctan2(Ix, Iy)
    return grad_mag_image, grad_dir_image


# create an edge magnitude image using the derivative operator
img_edge_mag, img_edge_dir = create_edge_magn_image(img, dx, dy)

# show all together
plt.subplot(121)
plt.imshow(img)
plt.axis('off')
plt.title('Original image')
plt.subplot(122)
plt.imshow(img_edge_mag)
plt.axis('off')
plt.title('Edge magnitude map')
plt.show()
