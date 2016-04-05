# #########################################################################
# (C) 2016 Elettra - Sincrotrone Trieste S.C.p.A.. All rights reserved.   #
#                                                                         #
# Copyright 2016. Elettra - Sincrotrone Trieste S.C.p.A. THE COMPANY      #
# ELETTRA - SINCROTRONE TRIESTE S.C.P.A. IS NOT REPONSIBLE FOR THE USE    #
# OF THIS SOFTWARE. If software is modified to produce derivative works,  #
# such modified software should be clearly marked, so as not to confuse   #
# it with the version available from Elettra Sincrotrone Trieste S.C.p.A. #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of Elettra - Sincotrone Trieste S.C.p.A nor      #
#       the names of its contributors may be used to endorse or promote   #
#       products derived from this software without specific prior        #
#       written permission.                                               #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY ELETTRA - SINCROTRONE TRIESTE S.C.P.A. AND #
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,  #
# BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND       #
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL      #
# ELETTRA - SINCROTRONE TRIESTE S.C.P.A. OR CONTRIBUTORS BE LIABLE FOR    #
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL  #
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE       #
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS           #
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER    #
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR         #
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF  #
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                              #
# #########################################################################

#
# Author: Francesco Brun
# Last modified: April, 4th 2016
#

from numpy import zeros, mean, median, var, copy

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
       
    Example (using tiffile.py)
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