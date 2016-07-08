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

from sys import argv, exit
from glob import glob
from os import linesep
from os.path import sep, basename, exists
from time import time
from numpy import float32, nanmin, nanmax
from multiprocessing import Process, Lock

from postprocess.postprocess import postprocess
from tifffile import imread, imsave



def main(argv):          
	"""To do...

	Usage
	-----
	

	Parameters
	---------
		   
	Example
	--------------------------
	The following line processes the first ten TIFF files of input path 
	"/home/in" and saves the processed files to "/home/out" with the 
	application of the Boin and Haibel filter with smoothing via a Butterworth
	filter of order 4 and cutoff frequency 0.01:

	reconstruct 0 4 C:\Temp\Dullin_Aug_2012\sino_noflat C:\Temp\Dullin_Aug_2012\sino_noflat\output 
	9.0 10.0 0.0 0.0 0.0 true sino slice C:\Temp\Dullin_Aug_2012\sino_noflat\tomo_conv flat dark

	"""	
	lock = Lock()
	# Get the from and to number of files to process:
	idx = int(argv[0])
	   
	# Get input and output paths:
	inpath = argv[1]
	outfile = argv[2]
	
	if not inpath.endswith(sep): inpath += sep

	# Get parameters:
	convert_opt = argv[3]
	crop_opt = argv[4]	
	crop_opt = '0:0:0:0'

	outprefix = argv[5]		
	logfilename = argv[6]	

	# Get the files in infile:		
	files = sorted(glob(inpath + '*.tif*'))
	num_files = len(files)		
	
	if ((idx >= num_files) or (idx == -1)):
		idx = num_files - 1

	# Read the image:
	im = imread(files[idx])

	# Process the image:		
	im = postprocess(im, convert_opt, crop_opt)	

	# Write down reconstructed preview file (file name modified with metadata):		
	im = im.astype(float32)
	outfile = outfile + '_' + str(im.shape[1]) + 'x' + str(im.shape[0]) + '_' + str( nanmin(im)) + '$' + str( nanmax(im) )	
	im.tofile(outfile)


if __name__ == "__main__":
	main(argv[1:])