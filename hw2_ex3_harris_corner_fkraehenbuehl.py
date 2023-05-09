""" 3 Corner detection """

# Imports
import numpy as np
import matplotlib.pyplot as plt
#import hw2_ex1_linear_filtering_template as ex1
from scipy.ndimage import gaussian_filter1d
from math import ceil, floor
from scipy import signal
from scipy import ndimage, misc
import numpy as np
from PIL import Image
import cv2

#import hw2_ex1_linear_filtering_fkraehenbuehl as ex1

from scipy import signal as sig
plt.rcParams['image.cmap'] = 'gray'
from scipy.signal import convolve2d, convolve
from skimage import color, io
import pdb
from scipy.ndimage import gaussian_filter



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
    offset = int(w_size/2)

    def gradient_x(imggray):
        ##Sobel operator kernels.
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        return sig.convolve2d(imggray, kernel_x, mode='same')

    def gradient_y(imggray):
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        return sig.convolve2d(imggray, kernel_y, mode='same')

    I_x = gradient_x(image)
    I_y = gradient_y(image)

    Ixx = gaussian_filter(I_x ** 2, sigma=sigma)
    Ixy = gaussian_filter(I_y * I_x, sigma=sigma)
    Iyy = gaussian_filter(I_y ** 2, sigma=sigma)
    #
    # # determinant
    # detA = Ixx * Iyy - Ixy ** 2
    # # trace
    # traceA = Ixx + Iyy
    #
    # R = detA - k * traceA ** 2
    Sxx=np.zeros((height,width))
    Sxy = np.zeros((height, width))
    Syy = np.zeros((height, width))
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            Sxx[y][x] = np.sum(Ixx[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Syy[y][x] = np.sum(Iyy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sxy[y][x] = np.sum(Ixy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
    # Find determinant and trace, use to get corner response
    det = (Sxx * Syy) - (Sxy ** 2)
    trace = Sxx + Syy
    R = det - k * (trace ** 2)

    return R

w_size=5
sigma=0.2
k=0.1

# 3.2
# Evaluate myharris on the image
R = myharris(img, w_size, sigma, k)
plt.imshow(R)
plt.colorbar()
plt.show()


# 3.3
# Repeat with rotated image by 45 degrees
# HINT: Use scipy.ndimage.rotate() function
angle=45
img_rotated = ndimage.rotate(img, angle)   ### your code should go here ##
R_rotated = myharris(img_rotated, w_size, sigma, k)
plt.imshow(R_rotated)
plt.colorbar()
plt.show()


# 3.4
# Repeat with downscaled image by a factor of half
# HINT: Use scipy.misc.imresize()scipy.misc.imresize() function
size=0.5
#img_scaled = misc.imresize(img,size)    ### your code should go here ###
img_scaled = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

R_scaled = myharris(img_scaled, w_size, sigma, k)
plt.imshow(R_scaled)
plt.colorbar()
plt.show()
