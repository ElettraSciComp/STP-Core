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
# Last modified: July, 17th 2016
#

from sys import argv, exit
from os import remove, sep,  linesep
from os.path import exists
from numpy import float32, nanmin, nanmax, isscalar

from preprocess.dynamic_flatfielding import dff_prepare_plan, dynamic_flat_fielding
from preprocess.extract_flatdark import extract_flatdark, _medianize

from h5py import File as getHDF5
import io.tdf as tdf

def main(argv):
          
	"""Preview the effects of dynamic flat fielding by processing the specified 
	projection of the input TDF file. A 32-bit RAW image is created as output.

    Parameters
    ----------
	argv[0] : int
		(Zero-order) index of the projection to process.

    argv[1] : string
		The absolute path of input TDF file to process.

	argv[2] : string
		The absolute path of output 32-bit RAW file that will be created by the 
		process. The input filename will be modified by appending width and height
		of the image to the string (min and max gray levels are also appended).

	argv[3] : int
		Downsampling parameter of the dynamic flat fielding (see reference).

	argv[4] : int
		Repetition parameter of the dynamic flat fielding (see reference).

    argv[5] : string
		The absolute path of a text file where error information is reported. 
		Note that only error information is reported. In case of a successful
		execution, nothing is reported in the log file. 

	References
	----------
	V. Van Nieuwenhove, J. De Beenhouwer, F. De Carlo, L. Mancini, F. Marone, 
	and J. Sijbers, "Dynamic intensity normalization using eigen flat fields 
	in X-ray imaging", Optics Express, 23(11), 27975-27989, 2015.

	Example
	-------
	The following line applies dynamic flat fielding to the 600-th projection of the 
	existing dataset S:\Temp\in.tdf	and it creates a new 32-bit RAW image with a file 
	name similar to S:\Temp\in_previewdff.raw_2048x2048_-0.1230087$1.002475:

	preview_dff.py 600 S:\Temp\in.tdf S:\Temp\in_previewdff.raw 2 10 R:\Temp\dff_log.txt
	
    """ 
	# Get the number of projection to pre-process:
	idx = int(argv[0])
	   
	# Get paths:
	infile = argv[1]
	outfile = argv[2]
	
	# Algorithm parameters:
	downs = int(argv[3])
	repet = int(argv[4])
		
	# Log file:
	logfilename = argv[5]		

	
	# Open the HDF5 file:	
	f_in = getHDF5(infile, 'r')
	
	skipflat = False
	skipdark = False
	try:
		if "/tomo" in f_in:
			dset = f_in['tomo']					
			if "/flat" in f_in:
				flat_dset = f_in['flat']
				if "/dark" in f_in:
					im_dark = _medianize(f_in['dark'])
				else:										
					skipdark = True
			else:
				skipflat = True # Nothing to do in this case			
		else: 
			dset = f_in['exchange/data']
			if "/exchange/data_white" in f_in:
				flat_dset = f_in['/exchange/data_white']
				if "/exchange/data_dark" in f_in:
					im_dark = _medianize(f_in['/exchange/data_dark'])
				else:					
					skipdark = True
			else:
				skipflat = True # Nothing to do in this case			
	except:
		skipflat = True
		log = open(logfilename,"a")
		log.write(linesep + "\tError reading input dataset. Process will end.")	
		log.close()			
		exit()

	# Check if the HDF5 makes sense:
	if (tdf.get_nr_projs(dset) == 0):
		log = open(logfilename,"a")
		log.write(linesep + "\tNo projections found. Process will end.")	
		log.close()			
		exit()
	
	if skipflat:
		log = open(logfilename,"a")
		log.write(linesep + "\tNo flat field images found. Process will end.")	
		log.close()			
		exit()

	if skipdark:
		tmp = tdf.read_tomo(dset,0)			
		im_dark = zeros(tmp.shape)
		log = open(logfilename,"a")
		log.write(linesep + "\tWarning: No dark field images found.")	
		log.close()

	# Prepare plan:
	EFF, filtEFF = dff_prepare_plan(flat_dset, repet, im_dark)	

	# Read input image:
	im = tdf.read_tomo(dset,idx).astype(float32)		
	f_in.close()
	
	# Apply dynamic flat fielding:
	im = dynamic_flat_fielding(im, EFF, filtEFF, downs, im_dark)
	
	# Write down reconstructed preview file (file name modified with metadata):		
	im = im.astype(float32)
	outfile = outfile + '_' + str(im.shape[1]) + 'x' + str(im.shape[0]) + '_' + str( nanmin(im)) + '$' + str( nanmax(im) )	
	im.tofile(outfile)

	
if __name__ == "__main__":
	main(argv[1:])
