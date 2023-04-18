""" 3 Corner detection """

# Imports
import numpy as np
import matplotlib.pyplot as plt
#import hw2_ex1_linear_filtering_template as ex1
from scipy.ndimage import gaussian_filter1d
from math import ceil, floor

import hw2_ex1_linear_filtering_fkraehenbuehl as ex1


plt.rcParams['image.cmap'] = 'gray'
from scipy.signal import convolve2d, convolve
from skimage import color, io
import pdb

# Load the image, convert to float and grayscale
#img = io.imread('chessboard.jpg')
img = io.imread('C:\\Users\\FlipFlop\\Documents\\UniBE\\Sem2_IntroToImageAnal\\hw2_handout\\chessboard.jpeg')
img = color.rgb2gray(img)

# 3.1
# Write a function myharris(image) which computes the harris corner for each pixel in the image. The function should return the R
# response at each location of the image.
# HINT: You may have to play with different parameters to have appropriate R maps.
# Try Gaussian smoothing with sigma=0.2, Gradient summing over a 5x5 region around each pixel and k = 0.1.)
def myharris(image, w_size, sigma, k):
    # This function computes the harris corner for each pixel in the image
    # INPUTS
    # @image    : a 2-D image as a numpy array
    # @w_size   : an integer denoting the size of the window over which the gradients will be summed
    # sigma     : gaussian smoothing sigma parameter
    # k         : harris corner constant
    # OUTPUTS
    # @R        : 2-D numpy array of same size as image, containing the R response for each image location

    ### your code should go here ###
    # compute derivatives and smooth
    # https://muthu.co/harris-corner-detector-implementation-in-python/
    height, width = image.shape
    dx = np.array([-1, 0, 1])
    dy = np.array([-1,0,1]).reshape(-1,1)#
    filter_length=3 #random
    sigma=1#temp to see something
    mygauss=ex1.gauss1d(sigma,filter_length)
    mygauss=np.asarray(mygauss)
    gdx = ex1.myconv2(dx,mygauss)
    gdy = ex1.myconv2(dy,mygauss)
    Ix = ex1.myconv2(image, gdx)
    Iy = ex1.myconv2(image, gdy)
    Ix = Ix[:height, :width]
    Iy = Iy[:height, :width]

    # product of gradients at each pixel
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # sums of products of derivates at each pixel
    Sxx = np.zeros((height, width))
    Syy = np.zeros((height, width))
    Sxy = np.zeros((height, width))
    offset = int(w_size / 2)  # to make sure windord doesnt get out of image
    height_offset=1
    width_offset=1

    for y in range(offset, height_offset):
        for x in range(offset, width_offset):
            Sxx[x, y] = np.sum(Ixx[x - offset:x + 1 + offset, y - offset:y + 1 + offset])
            Syy[x, y] = np.sum(Iyy[x - offset:x + 1 + offset, y - offset:y + 1 + offset])
            Sxy[x, y] = np.sum(Ixy[x - offset:x + 1 + offset, y - offset:y + 1 + offset])

    # Define at each pixel matrix, compute response
    det = (Sxx * Syy) - (Sxy ** 2)
    trace = Sxx + Syy
    r = det - k * (trace ** 2)
    R=r
    return R
# w_size=5
# sigma=0.2
# k=0.1
# image=img
# myharris(image, w_size, sigma, k)

# 3.2
# Evaluate myharris on the image
R = myharris(img, 5, 0.2, 0.1)
plt.imshow(R)
plt.colorbar()
plt.show()
pass

# 3.3
# Repeat with rotated image by 45 degrees
# HINT: Use scipy.ndimage.rotate() function
#R_rotated =     ### your code should go here ###
# plt.imshow(R_rotated)
# plt.colorbar()
# plt.show()


# 3.4
# Repeat with downscaled image by a factor of half
# HINT: Use scipy.misc.imresize() function
#R_scaled =      ### your code should go here ###
# plt.imshow(R_scaled)
# plt.colorbar()
# plt.show()
