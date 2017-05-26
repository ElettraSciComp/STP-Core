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
# Last modified: August, 8th 2016
#

from sys import argv, exit
from os import remove, sep,  linesep
from os.path import exists
from numpy import float32, float16, amin, amax, isscalar, sqrt, max, empty_like, pad
from time import time
from multiprocessing import Process, Lock

from postprocess.polarfilters.homomorphic import homomorphic
from preprocess.ringremoval.raven import raven

from h5py import File as getHDF5
import stpio.tdf as tdf
import cv2

from tifffile import imread, imsave



def main(argv):  
			
	im = imread('C:\\Users\\Franz\\Documents\\MATLAB\\dering\\test2.tif')
	
	# Get original size:
	origsize = im.shape

	# Up-scaling:
	im = cv2.resize(im, None, 1.0, 1.0, cv2.INTER_CUBIC)
	rows, cols = im.shape
	cen_x = im.shape[1] / 2
	cen_y = im.shape[0] / 2

	# To polar:
	im = cv2.linearPolar(im, (cen_x, cen_y), amax([rows,cols]), cv2.INTER_CUBIC)

	# Padding:
	#cropsize = im.shape
	#im = pad(im, ((origsize[0] / 4, origsize[0] / 4), (origsize[1] / 2, 0)), 'symmetric')    
	#imsave('C:\\Temp\\padded.tif', im)

	# Filter:
	#im = homomorphic(im, "0.95;0.15")

	im = raven(im, "5;0.95")

	# Crop:
	#im = im[origsize[0] / 4:origsize[0] / 4 + cropsize[0],origsize[1] / 2:origsize[1] / 2 + cropsize[1]]
	#imsave('C:\\Temp\\cropped.tif', im)

	# Back to cartesian:
	im = cv2.linearPolar(im, (cen_x, cen_y), amax([rows,cols]), cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
	imsave('C:\\Temp\\p2c.tif', im)	

	# Down-scaling to original size:
	im = cv2.resize(im, (origsize[0], origsize[1]), interpolation = cv2.INTER_CUBIC)
	imsave('C:\\Temp\\output.tif', im)

	
if __name__ == "__main__":
	main(argv[1:])
