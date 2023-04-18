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
dx = [-1,0,1]### your code should go here ###
dy = [-1,0,1]### your code should go here ###

# convolve derivative operator with a 1d gaussian filter with sigma = 1
# You should end up with 2 1d edge filters,  one identifying edges in the x direction, and
# the other in the y direction
sigma = 1
filterlenght=3
### your code should go here ###
result=ex1.gauss1d(sigma,filterlenght)
result=np.asarray(result)
dx=np.asarray(dx)
result=ex1.myconv2(dx,result)

print("dx",ex1.myconv2(dx,np.array(ex1.mygauss(sigma,filterlenght))))
print("dy",ex1.myconv2(dy,ex1.mygauss(sigma,filterlenght)))


# gdx = ### your code should go here ###
# gdy = ### your code should go here ###


# 2.2
# Gradient Edge Magnitude Map
def create_edge_magn_image(image, dx, dy):
    # this function created an eddge magnitude map of an image
    # for every pixel in the image, it assigns the magnitude of gradients
    # INPUTS
    # @image  : a 2D image
    # @gdx     : gradient along x axis
    # @gdy     : gradient along y axis
    # OUTPUTS
    # @ grad_mag_image  : 2d image same size as image, with the magnitude of gradients in every pixel
    # @grad_dir_image   : 2d image same size as image, with the direcrion of gradients in every pixel

    ### your code should go here ###

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
