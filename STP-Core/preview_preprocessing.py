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

from sys import argv, exit
from os import remove, sep,  linesep
from os.path import exists
from numpy import float32, nanmin, nanmax
from time import time
from multiprocessing import Process, Lock

from preprocess.extfov_correction import extfov_correction
from preprocess.flat_fielding import flat_fielding
from preprocess.ring_correction import ring_correction
from preprocess.extract_flatdark import extract_flatdark

from h5py import File as getHDF5
from utils.caching import cache2plan, plan2cache
import io.tdf as tdf

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

	destripe /home/in /home/out 1 10 1 0.01 4    

	"""
	lock = Lock()

	skip_ringrem = False
	skip_flat = False
	skip_flat_after = True
	first_done = False	

	# Get the number of sino to pre-process:
	idx = int(argv[0])
	   
	# Get paths:
	infile = argv[1]
	outfile = argv[2]
	
	# Normalization parameters:
	norm_sx = int(argv[3])
	norm_dx = int(argv[4])
	
	# Params for flat fielding with post flats/darks:
	flat_end = True if argv[5] == "True" else False
	half_half = True if argv[6] == "True" else False
	half_half_line = int(argv[7])
		
	# Params for extended FOV:
	ext_fov = True if argv[8] == "True" else False
	ext_fov_rot_right = argv[9]
	if ext_fov_rot_right == "True":
		ext_fov_rot_right = True
		if (ext_fov):
			norm_sx = 0
	else:
		ext_fov_rot_right = False
		if (ext_fov):
			norm_dx = 0		
	ext_fov_overlap = int(argv[10])
		
	# Method and parameters coded into a string:
	ringrem = argv[11]	
	
	# Tmp path and log file:
	tmppath = argv[12]	
	if not tmppath.endswith(sep): tmppath += sep		
	logfilename = argv[13]		

	
	# Open the HDF5 file:	
	f_in = getHDF5(infile, 'r')
	if "/tomo" in f_in:
		dset = f_in['tomo']		
	else: 
		dset = f_in['exchange/data']
		prov_dset = f_in['provenance/detector_output']			
		
	num_proj = tdf.get_nr_projs(dset)
	num_sinos = tdf.get_nr_sinos(dset)
	
	# Check if the HDF5 makes sense:
	if (num_sinos == 0):
		log = open(logfilename,"a")
		log.write(linesep + "\tNo projections found. Process will end.")	
		log.close()			
		exit()		

	# Get flat and darks from cache or from file:
	try:
		corrplan = cache2plan(infile, tmppath)
	except Exception as e:
		#print "Error(s) when reading from cache"
		corrplan = extract_flatdark(f_in, flat_end, logfilename)
		plan2cache(corrplan, infile, tmppath)		

	# Read input image:
	im = tdf.read_sino(dset,idx).astype(float32)		
	f_in.close()	

	# Perform pre-processing (flat fielding, extended FOV, ring removal):	
	im = flat_fielding(im, idx, corrplan, flat_end, half_half, half_half_line, norm_sx, norm_dx)			
	im = extfov_correction(im, ext_fov, ext_fov_rot_right, ext_fov_overlap)
	im = ring_correction (im, ringrem, flat_end, corrplan['skip_flat_after'], half_half, half_half_line, ext_fov)							
	
	# Write down reconstructed preview file (file name modified with metadata):		
	im = im.astype(float32)
	outfile = outfile + '_' + str(im.shape[1]) + 'x' + str(im.shape[0]) + '_' + str( nanmin(im)) + '$' + str( nanmax(im) )	
	im.tofile(outfile)

	
if __name__ == "__main__":
	main(argv[1:])
