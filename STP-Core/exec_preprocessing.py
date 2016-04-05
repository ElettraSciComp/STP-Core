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
from numpy import float32, amin, amax
from time import time
from multiprocessing import Process, Lock

from preprocess.extfov_correction import extfov_correction
from preprocess.flat_fielding import flat_fielding
from preprocess.ring_correction import ring_correction
from preprocess.extract_flatdark import extract_flatdark

from h5py import File as getHDF5
import io.tdf as tdf


def _write_data(lock, im, index, outfile, outshape, outtype, logfilename, cputime, itime):    	      

	lock.acquire()
	try:        
		t0 = time() 			
		f_out = getHDF5( outfile, 'a' )					 
		f_out_dset = f_out.require_dataset('exchange/data', outshape, outtype, chunks=tdf.get_dset_chunks(outshape[0])) 
		tdf.write_sino(f_out_dset,index,im.astype(float32))
					
		# Set minimum and maximum:
		if ( amin(im[:]) < float(f_out_dset.attrs['min']) ):
			f_out_dset.attrs['min'] = str(amin(im[:]))
		if ( amax(im[:]) > float(f_out_dset.attrs['max'])):
			f_out_dset.attrs['max'] = str(amax(im[:]))		
		f_out.close()			
		t1 = time() 

		# Print out execution time:
		log = open(logfilename,"a")
		log.write(linesep + "\tsino_%s processed (CPU: %0.3f sec - I/O: %0.3f sec)." % (str(index).zfill(4), cputime, t1 - t0 + itime))
		log.close()	

	finally:
		lock.release()	

def _process (lock, int_from, int_to, infile, outfile, outshape, outtype, plan, norm_sx, norm_dx, flat_end, 
			 half_half, half_half_line, ext_fov, ext_fov_rot_right, ext_fov_overlap, ringrem, logfilename):

	# Process the required subset of images:
	for i in range(int_from, int_to + 1):                 
				
		# Read input image:
		t0 = time()
		f_in = getHDF5(infile, 'r')
		if "/tomo" in f_in:
			dset = f_in['tomo']
		else: 
			dset = f_in['exchange/data']
		im = tdf.read_sino(dset,i).astype(float32)		
		f_in.close()
		t1 = time() 		

		# Perform pre-processing (flat fielding, extended FOV, ring removal):	
		im = flat_fielding(im, i, plan, flat_end, half_half, half_half_line, norm_sx, norm_dx)			
		im = extfov_correction(im, ext_fov, ext_fov_rot_right, ext_fov_overlap)
		im = ring_correction (im, ringrem, flat_end, plan['skip_flat_after'], half_half, half_half_line, ext_fov)
		t2 = time() 		
								
		# Save processed image to HDF5 file (atomic procedure - lock used):
		_write_data(lock, im, i, outfile, outshape, outtype, logfilename, t2 - t1, t1 - t0)


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

	# Get the from and to number of files to process:
	int_from = int(argv[0])
	int_to = int(argv[1])
	   
	# Get paths:
	infile = argv[2]
	outfile = argv[3]
	
	# Normalization parameters:
	norm_sx = int(argv[4])
	norm_dx = int(argv[5])
	
	# Params for flat fielding with post flats/darks:
	flat_end = True if argv[6] == "True" else False
	half_half = True if argv[7] == "True" else False
	half_half_line = int(argv[8])
		
	# Params for extended FOV:
	ext_fov = True if argv[9] == "True" else False
	ext_fov_rot_right = argv[10]
	if ext_fov_rot_right == "True":
		ext_fov_rot_right = True
		if (ext_fov):
			norm_sx = 0
	else:
		ext_fov_rot_right = False
		if (ext_fov):
			norm_dx = 0		
	ext_fov_overlap = int(argv[11])
		
	# Method and parameters coded into a string:
	ringrem = argv[12]	
	
	# Nr of threads and log file:
	nr_threads = int(argv[13])
	logfilename = argv[14]		




	# Log input parameters:
	log = open(logfilename,"w")
	log.write(linesep + "\tInput TDF file: %s" % (infile))	
	log.write(linesep + "\tOutput TDF file: %s" % (outfile))		
	log.write(linesep + "\t--------------")	
	log.write(linesep + "\tOpening input dataset...")	
	log.close()
	
	# Remove a previous copy of output:
	if exists(outfile):
		remove(outfile)
	
	# Open the HDF5 file:	
	f_in = getHDF5(infile, 'r')
	if "/tomo" in f_in:
		dset = f_in['tomo']
		
		tomoprefix = 'tomo'
		flatprefix = 'flat'
		darkprefix = 'dark'
	else: 
		dset = f_in['exchange/data']
		prov_dset = f_in['provenance/detector_output']		
		
		tomoprefix = prov_dset.attrs['tomo_prefix']
		flatprefix = prov_dset.attrs['flat_prefix']
		darkprefix = prov_dset.attrs['dark_prefix']
		
		
	num_proj = tdf.get_nr_projs(dset)
	num_sinos = tdf.get_nr_sinos(dset)
	
	if (num_sinos == 0):
		log = open(logfilename,"a")
		log.write(linesep + "\tNo projections found. Process will end.")	
		log.close()			
		exit()		

	# Check extrema (int_to == -1 means all files):
	if ((int_to >= num_sinos) or (int_to == -1)):
		int_to = num_sinos - 1

	# Prepare the working plan for flat and dark images:
	log = open(logfilename,"a")
	log.write(linesep + "\t-------")
	log.write(linesep + "\tPreparing the working plan...")				
	log.close()

	# Extract flat and darks:
	plan = extract_flatdark(f_in, flat_end, logfilename)
	
	# Outfile shape can be determined only after first processing in ext FOV mode:
	if (ext_fov):

		# Read input sino:
		idx = num_sinos / 2
		im = tdf.read_sino(dset,idx).astype(float32)		
	
		# Perform pre-processing (flat fielding, extended FOV, ring removal):	
		im = flat_fielding(im, idx, plan, flat_end, half_half, half_half_line, norm_sx, norm_dx)			
		im = extfov_correction(im, ext_fov, ext_fov_rot_right, ext_fov_overlap)
		im = ring_correction (im, ringrem, flat_end, plan['skip_flat_after'], half_half, half_half_line, ext_fov)					
	
		# Write down reconstructed preview file (file name modified with metadata):		
		outshape = tdf.get_dset_shape(im.shape[1], num_sinos, im.shape[0])			
		f_out = getHDF5(outfile, 'w')
		f_out_dset = f_out.create_dataset('exchange/data', outshape, im.dtype, chunks=tdf.get_dset_chunks(im.shape[1])) 
		f_out_dset.attrs['min'] = str(amin(im[:]))
		f_out_dset.attrs['max'] = str(amax(im[:]))
		f_out.close()

	else:
		im = tdf.read_tomo(dset,0).astype(float32)	
		outshape = tdf.get_dset_shape(im.shape[1], im.shape[0], num_proj)			
		f_out = getHDF5(outfile, 'w')
		f_out_dset = f_out.create_dataset('exchange/data', outshape, im.dtype, chunks=tdf.get_dset_chunks(im.shape[1])) 
		f_out_dset.attrs['min'] = str(amin(im[:]))
		f_out_dset.attrs['max'] = str(amax(im[:]))
		f_out.close()
	
	f_in.close()
		
	# Log infos:
	log = open(logfilename,"a")
	log.write(linesep + "\tWorking plan prepared correctly.")	
	log.write(linesep + "\t-------")
	log.write(linesep + "\tPerforming pre processing...")			
	log.close()	

	# Run several threads for independent computation without waiting for threads completion:
	for num in range(nr_threads):
		start = (num_sinos / nr_threads)*num
		if (num == nr_threads - 1):
			end = num_sinos - 1
		else:
			end = (num_sinos / nr_threads)*(num + 1) - 1
		Process(target=_process, args=(lock, start, end, infile, outfile, outshape, im.dtype, plan, norm_sx, norm_dx, flat_end, 
								half_half, half_half_line, ext_fov, ext_fov_rot_right, ext_fov_overlap, ringrem, logfilename )).start()


	#start = 0
	#end = num_sinos - 1
	#process(lock, start, end, infile, outfile, outshape, im.dtype, plan, norm_sx, norm_dx, flat_end, 
	#		half_half, half_half_line, ext_fov, ext_fov_rot_right, ext_fov_overlap, ringrem, logfilename)

	# 255 256 C:\Temp\BrunGeorgos.tdf C:\Temp\BrunGeorgos_corr.tdf 11 11 True True 900 False False 0 rivers:11;0 1 C:\Temp\log_00.txt

	
if __name__ == "__main__":
	main(argv[1:])
