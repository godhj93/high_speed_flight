from xml.dom.minicompat import NodeList
import numpy as np 
import cv2 
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy import ndimage


def add_gaussian_shifts(depth, focal_length,baseline_m, std=0.0):

    rows, cols = depth.shape 
    #gaussian_shifts = np.random.normal(0, std, size=(rows, cols, 2))
    
    gaussian_shifts=np.zeros([rows, cols,2])
    
    for i in range(rows):
        for j in range(cols):
            xx=abs(i-(rows/2))
            xy=abs(j-(cols/2)) 
            theta=np.math.atan(np.math.sqrt(pow(xx+xy,2))/focal_length)
            sigma= (0.8+0.035*(theta/(np.math.pi/2-theta)))*depth[i,j]*100*(1/focal_length)
            # if ((i==300) and (j==300)):
            #     print(sigma)
            gaussian_shifts[i,j,0]=np.random.normal(0,sigma)
            gaussian_shifts[i,j,1]=np.random.normal(0,sigma)
    
    
    gaussian_shifts = gaussian_shifts.astype(np.float32)

    # creating evenly spaced coordinates  
    xx = np.linspace(0, cols-1, cols)
    yy = np.linspace(0, rows-1, rows)

    # get xpixels and ypixels 
    xp, yp = np.meshgrid(xx, yy)

    xp = xp.astype(np.float32)
    yp = yp.astype(np.float32)

    xp_interp = np.minimum(np.maximum(xp + gaussian_shifts[:, :, 0], 0.0), cols)
    yp_interp = np.minimum(np.maximum(yp + gaussian_shifts[:, :, 1], 0.0), rows)

    depth_interp = cv2.remap(depth, xp_interp, yp_interp, cv2.INTER_LINEAR)

    return depth_interp
    

def filterDisp(disp, dot_pattern_, invalid_disp_):

    size_filt_ = 9

    xx = np.linspace(0, size_filt_-1, size_filt_)
    yy = np.linspace(0, size_filt_-1, size_filt_)

    xf, yf = np.meshgrid(xx, yy)

    xf = xf - int(size_filt_ / 2.0)
    yf = yf - int(size_filt_ / 2.0)

    sqr_radius = (xf**2 + yf**2)
    vals = sqr_radius * 1.2**2 

    vals[vals==0] = 1 
    weights_ = 1 /vals  

    fill_weights = 1 / ( 1 + sqr_radius)
    fill_weights[sqr_radius > 9] = -1.0 

    disp_rows, disp_cols = disp.shape 
    dot_pattern_rows, dot_pattern_cols = dot_pattern_.shape

    lim_rows = np.minimum(disp_rows - size_filt_, dot_pattern_rows - size_filt_)
    lim_cols = np.minimum(disp_cols - size_filt_, dot_pattern_cols - size_filt_)

    center = int(size_filt_ / 2.0)

    window_inlier_distance_ = 0.1

    out_disp = np.ones_like(disp) * invalid_disp_

    interpolation_map = np.zeros_like(disp)

    for r in range(0, lim_rows):

        for c in range(0, lim_cols):

            if dot_pattern_[r+center, c+center] > 0:
                                
                # c and r are the top left corner 
                window  = disp[r:r+size_filt_, c:c+size_filt_] 
                dot_win = dot_pattern_[r:r+size_filt_, c:c+size_filt_] 
  
                valid_dots = dot_win[window < invalid_disp_]

                n_valids = np.sum(valid_dots) / 255.0 
                n_thresh = np.sum(dot_win) / 255.0 

                if n_valids > n_thresh / 1.2: 

                    mean = np.mean(window[window < invalid_disp_])

                    diffs = np.abs(window - mean)
                    diffs = np.multiply(diffs, weights_)

                    cur_valid_dots = np.multiply(np.where(window<invalid_disp_, dot_win, 0), 
                                                 np.where(diffs < window_inlier_distance_, 1, 0))

                    n_valids = np.sum(cur_valid_dots) / 255.0

                    if n_valids > n_thresh / 1.2: 
                    
                        accu = window[center, center] 

                        assert(accu < invalid_disp_)

                        out_disp[r+center, c + center] = round((accu)*8.0) / 8.0

                        interpolation_window = interpolation_map[r:r+size_filt_, c:c+size_filt_]
                        disp_data_window     = out_disp[r:r+size_filt_, c:c+size_filt_]

                        substitutes = np.where(interpolation_window < fill_weights, 1, 0)
                        interpolation_window[substitutes==1] = fill_weights[substitutes ==1 ]

                        disp_data_window[substitutes==1] = out_disp[r+center, c+center]

    return out_disp


def make_axial_noise(depth,max,min):
    h, w = depth.shape 
    
    for i in range(h):
        for j in range(w):
            z=depth[i,j];
            sigma=(9*pow(z,2)-26.5*z+20.237)*0.001
            depth[i,j]=depth[i,j]+np.random.normal(0,sigma)
            if(depth[i,j]>max):
                depth[i,j]=0
                
            if(depth[i,j]<min):
                depth[i,j]=0
    
    return depth

def make_blurr(image,size,angle):
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size, dtype=np.float32)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )  
    kernel=kernel*(1.0/np.sum(kernel))

     
    
    output=cv2.filter2D(image,-1,kernel)

    
    return output
    

def make_noisy_image(image,dot_pattern,focal_length,baseline_m,range_max,range_min,kernel_size,blurr_angle):
    scale_factor  = 100
    invalid_disp_ = 99999999.9
    
    h, w = 256,256#image.shape
    blurr_image=make_blurr(image,kernel_size,blurr_angle)
    depth =  blurr_image.astype('float')/255.0*10
    depth_interp = add_gaussian_shifts(depth, focal_length,baseline_m)
    
    disp_= focal_length * baseline_m / (depth_interp + 1e-10)
    depth_f = np.round(disp_ * 8.0)/8.0
    
    out_disp = filterDisp(depth_f, dot_pattern, invalid_disp_)
    depth = focal_length * baseline_m / out_disp
    depth[out_disp == invalid_disp_] = 0
    
    noisy_depth = make_axial_noise(depth,range_max,range_min)
    # noisy_depth = noisy_depth * 255.0/10
    # noisy_depth = noisy_depth.astype('uint8')
    
    return noisy_depth
    
    