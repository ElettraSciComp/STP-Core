###########################################################################
# (C) 2016 Elettra - Sincrotrone Trieste S.C.p.A.. All rights reserved.   #
#                                                                         #
#                                                                         #
# This file is part of STP-Core, the Python core of SYRMEP Tomo Project,  #
# a software tool for the reconstruction of experimental CT datasets.     #
#                                                                         #
# STP-Core is free software: you can redistribute it and/or modify it     #
# under the terms of the GNU General Public License as published by the   #
# Free Software Foundation, either version 3 of the License, or (at your  #
# option) any later version.                                              #
#                                                                         #
# STP-Core is distributed in the hope that it will be useful, but WITHOUT #
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or   #
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License    #
# for more details.                                                       #
#                                                                         #
# You should have received a copy of the GNU General Public License       #
# along with STP-Core. If not, see <http://www.gnu.org/licenses/>.        #
#                                                                         #
###########################################################################

#
# Author: Francesco Brun
# Last modified: July, 8th 2016
#

from numpy import zeros, mean, median, var, copy, vstack

def sijberspostnov(im, args):
    """Process a sinogram image with the Sijbers and Postnov de-striping algorithm.

    Parameters
    ----------
    im : array_like
        Image data as numpy array. 
   
    winsize : int
        Size of the local floating window used to look for homogeneity.

    thresh : float
        Image rows (within the floating window) having variance below 
        this tresh will be corrected.
       
    Example
    --------------------------
    >>> im = imread('original.tif')
    >>> im = sijberspostnov_filter(im, 51, 0.001)    
    >>> imsave('filtered.tif', im) 

    References
    ----------
    J. Sijbers and A. Postnov, Reduction of ring artifacts in high resolution
    micro-CT reconstructions, Physics in Medicine and Biology 49(14):247-253, 2004.

    """  

    # Initializations:
    dimx = im.shape[1]
    dimy = im.shape[0]
    
     # Get args:
    winsize, thresh  = args.split(";")     
    winsize = int(winsize)
    thresh  = float(thresh)

    # Normalize thresh parameter:
    #thresh = thresh*65536.0
        
    glob_art = zeros(dimx)
    prevsize = 0

    # Within a sliding window:
    for i in range(0, dimx - winsize):
            
        ct = 0
        matrix = zeros(winsize)

        # For each line of the current window:
        for j in range(0, dimy):
        
            # Compute the variance within current sliding window:
            v = im[j, i:(i + winsize)]            
            curr_var = var(v)
        
            # If variance is below threshold:
            if (curr_var < thresh): 
                # Add current line with mean subtracted to a temporary matrix:
                v = v - mean(v)   
                matrix = vstack([matrix,v])    
                ct = ct + 1
            
        # Determine local artifact correction vector:
        if (ct > 1):
            ct = ct - 1
            matrix = matrix[1:ct,:]
            loc_art = median(matrix, axis=0)
        else:
            if (ct == 1):
                loc_art = matrix[1,:]
            else:
                loc_art = zeros(winsize)
           
        # Determine global artifact correction vector:
        for k in range(0, winsize):
            if (matrix.shape[0] > prevsize):
                glob_art[k + i] = loc_art[k]             
    
        prevsize = matrix.shape[0]    

    # Correct each line of the input image:
    for i in range(0, im.shape[0]):
        im[i,:] = im[i,:] - glob_art

     # Return image according to input type:
    if (im.dtype == 'uint16'):
        
        # Check extrema for uint16 images:
        im[im < iinfo(uint16).min] = iinfo(uint16).min
        im[im > iinfo(uint16).max] = iinfo(uint16).max

        # Return image:
        return im.astype(uint16)
    else:
        return im