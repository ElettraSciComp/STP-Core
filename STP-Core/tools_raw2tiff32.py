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

import os
import os.path

from sys import argv, exit
from tifffile import *
from numpy import zeros, fromfile, float32

def main(argv):    
	"""Convert an input 32-bit RAW image to TIFF format

    Parameters
    ----------
    argv[0] : string
		The absolute path of input 32-bit RAW image file.

	argv[1] : string
		The absolute path of output 32-bit TIFF image file.

	argv[2] : int
		Width of the input RAW image.

	argv[3] : int
		Height of the input RAW image.

	Example
	-------
	tools_raw2tiff32 "R:\\slice.raw" "R:\\slice.tiff" 2048 2048
	
    """
	#
	# Get the parameters:
	#
	infile  = argv[0]
	outfile = argv[1]
	width   = int(argv[2]) 
	height  = int(argv[3]) 
	
	#
	# Body
	#	
	
	# Check if file exists:
	if not os.path.exists(infile):		
		exit()	
	
	try:
		# Prepare RAW matrix:
		im = zeros((width,height), dtype=float32)
		
		# Read RAW file:
		im = fromfile(infile, float32).reshape((height,width))
		
		# Save TIFF 32:
		imsave(outfile, im)				
	
	except:	
		exit()

if __name__ == "__main__":
	main(argv[1:])
