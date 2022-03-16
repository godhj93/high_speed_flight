import numpy as np 
import cv2 
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy import ndimage

from math import pi

import make_noisy_image


if __name__ == "__main__":
    dot_pattern_ = cv2.imread("./data/kinect-pattern_3x3.png", 0)
    
    focal_length  = 582.7  # focal length of the camera used 
    baseline_m    = 0.075
    range_max=10;
    range_min=0;
    kernel_size=1
    blurr_angle=45
    
    count=21
    image= cv2.imread("depth/{}.png".format(count), cv2.IMREAD_UNCHANGED)
    
    noisy_image=make_noisy_image.make_noisy_image(image,dot_pattern_,focal_length,baseline_m,range_max,range_min,kernel_size,blurr_angle)
    
    
    plt.imshow(noisy_image)
    plt.show()
    

    #plt.pcolormesh(noisy_image)
    #plt.colorbar()
    #plt.show()
    cv2.imwrite('noisy_{}.png'.format(count),noisy_image)