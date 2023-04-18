""" 1 Linear filtering """

# Imports
from math import sqrt, exp

import numpy as np
import matplotlib.pyplot as plt
from mpmath import pi
from scipy import signal
#from scipy import datasets
from scipy.ndimage import gaussian_filter1d

plt.rcParams['image.cmap'] = 'gray'
import time
import pdb
img = plt.imread('cat.jpg').astype(np.float32)
plt.imshow(img)
plt.axis('off')
plt.title('original image')
plt.show()
# 1.1
def boxfilter(n):
    # this function returns a box filter of size nxn

    ### your code should go here ###
    box_filter = 1 / n * np.ones((n, n))
    return box_filter

# 1.2
# Implement full convolution
def myconv2(myimage, myfilter):
    # This function performs a 2D convolution between image and filt, image being a 2D image. This
    # function should return the result of a 2D convolution of these two images. DO
    # NOT USE THE BUILT IN SCIPY CONVOLVE within this function. You should code your own version of the
    # convolution, valid for both 2D and 1D filters.
    # INPUTS
    # @ image         : 2D image, as numpy array, size mxn
    # @ filt          : 1D or 2D filter of size kxl
    # OUTPUTS
    # img_filtered    : 2D filtered image, of size (m+k-1)x(n+l-1)

    ### your code should go here ###
    m, n = myimage.shape
    k, l = myfilter.shape
    #https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
    #https://www.allaboutcircuits.com/technical-articles/two-dimensional-convolution-in-image-processing/

    #2D case
    newfilter=np.flipud(np.fliplr(myfilter))#flip filter downwards and leftwards
    newimage=np.pad(myimage,max(k-1,l-1))#zeropadding
    img_filtered=np.zeros((m+k-1, n+l-1))
    for i in range(m + k - 1):
        for j in range(n + l - 1):
            img_filtered[i, j] = np.sum(newfilter * newimage[i:i + k, j:j + l])
    return img_filtered


# 1.3
# create a boxfilter of size 10 and convolve this filter with your image - show the result
bsize = 10


### your code should go here ###


# 1.4
# create a function returning a 1D gaussian kernel
def gauss1d(sigma, filter_length=20):
    # INPUTS
    # @ sigma         : sigma of gaussian distribution
    # @ filter_length : integer denoting the filter length, default is 10
    # OUTPUTS
    # @ gauss_filter  : 1D gaussian filter

    ### your code should go here ###
    #https: // stackoverflow.com / questions / 11209115 / creating - gaussian - filter - of - required - length - in -python
    if filter_length%2==0:
        #even
        r = range(-int(filter_length / 2), int(filter_length / 2))
    else:
        #odd
        r = range(-int(filter_length / 2), int(filter_length / 2) + 1)
    print("r",r)
    gauss_filter=[1 / (sigma * sqrt(2 * pi)) * exp(-float(x) ** 2 / (2 * sigma ** 2)) for x in r]

    return gauss_filter

### your code should go here ###

# 1.1
# N=3
# print("boxfilter",boxfilter(N))

# 1.2
#2D-case
m,n,k,l=10,10,3,3
myimage = np.random.random((m,n))
myfilter = np.random.random((k,l))
myresult = myconv2(myimage, myfilter)
inbuiltresult = signal.convolve2d(myimage, myfilter, boundary='fill', mode='full')
if np.isclose(myresult, inbuiltresult).all:
    print("2D case works")
else:
    print("2D case doesn't work")
#1D-case, (1,3)
m,n,k,l=10,10,1,3
myimage = np.random.random((m,n))
myfilter = np.random.random((k,l))
myresult = myconv2(myimage, myfilter)
inbuiltresult = signal.convolve2d(myimage, myfilter, boundary='fill', mode='full')
if np.isclose(myresult, inbuiltresult).all:
    print("(1,3)-1D filter case works")
else:
    print("(1,3)-1D filter case doesn't work")
#1D-case, (3,1)
m,n,k,l=10,10,3,1
myimage = np.random.random((m,n))
myfilter = np.random.random((k,l))
myresult = myconv2(myimage, myfilter)
inbuiltresult = signal.convolve2d(myimage, myfilter, boundary='fill', mode='full')
if np.isclose(myresult, inbuiltresult).all:
    print("(3,1)-1D filter case works")
else:
    print("(3,1)-1D filter case doesn't work")

# 1.2
newimg=myconv2(img,boxfilter(11))
plt.imshow(newimg)
plt.axis('off')
plt.title('new image')
plt.show()
sigma, filter_length = 1, 11
mygauss=gauss1d(sigma,filter_length)
print("mygauss",mygauss)
#comparision to built-in
inbuiltresult=gaussian_filter1d(np.ones(filter_length),sigma)
if np.isclose(mygauss, inbuiltresult).all:
    print("mygauss works")
else:
    print("(mygauss doesn't work")