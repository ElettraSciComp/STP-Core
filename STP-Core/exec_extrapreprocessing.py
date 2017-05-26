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
from numpy import float32, float16, amin, amax, isscalar
from time import time
from multiprocessing import Process, Lock

from preprocess.extfov_correction import extfov_correction
from preprocess.flat_fielding import flat_fielding
from preprocess.dynamic_flatfielding_projections import dff_prepare_plan, dynamic_flat_fielding
from preprocess.ring_correction import ring_correction
from preprocess.extract_flatdark import extract_flatdark, _medianize

from h5py import File as getHDF5
import stpio.tdf as tdf
import cv2


def _write_data(lock, im, index, outfile, outshape, outtype, rescale_factor, logfilename, cputime, itime):    	      

	lock.acquire()
	try:        
		t0 = time() 			
		f_out = getHDF5(outfile, 'a')					 
		f_out_dset = f_out.require_dataset('exchange/data', outshape, outtype, chunks=tdf.get_dset_chunks(outshape[0])) 
		im = im * rescale_factor
		tdf.write_tomo(f_out_dset,index,im.astype(outtype))
					
		# Set minimum and maximum:
		if (amin(im[:]) < float(f_out_dset.attrs['min'])):
			f_out_dset.attrs['min'] = str(amin(im[:]))
		if (amax(im[:]) > float(f_out_dset.attrs['max'])):
			f_out_dset.attrs['max'] = str(amax(im[:]))		
		f_out.close()			
		t1 = time() 

		# Print out execution time:
		log = open(logfilename,"a")
		log.write(linesep + "\ttomo_%s processed (CPU: %0.3f sec - I/O: %0.3f sec)." % (str(index).zfill(4), cputime, t1 - t0 + itime))
		log.close()	

	finally:
		lock.release()	

def _process(lock, int_from, int_to, infile, outfile, outshape, outtype, skipflat, plan, flat_end, 
			 half_half, half_half_line, dynamic_ff, EFF, filtEFF, im_dark, rotation, interp, border, 
			 rescale_factor, logfilename):

	# Process the required subset of images:
	for i in range(int_from, int_to + 1):                 
				
		# Read input image:
		t0 = time()
		f_in = getHDF5(infile, 'r')
		if "/tomo" in f_in:
			dset = f_in['tomo']
		else: 
			dset = f_in['exchange/data']
		im = tdf.read_tomo(dset,i).astype(float32)		
		f_in.close()
		t1 = time() 		

		# Perform pre-processing (flat fielding, extended FOV, ring removal):
		if not skipflat:
			if dynamic_ff:
				# Dynamic flat fielding with downsampling = 2:
				im = dynamic_flat_fielding(im, EFF, filtEFF, 2, im_dark)
			else:
				im = flat_fielding(im, i, plan, flat_end, half_half, half_half_line, norm_sx, norm_dx)			
		t2 = time() 		

		# Rotate:
		rows, cols = im.shape
		#if (cols > rows):
		#	angle = - arctan(2.0 / cols) * (rows / 2.0) / 2.0
		#else:
		#	angle = - arctan(2.0 / rows) * (cols / 2.0) / 2.0
		#print(angle)
	
		M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)

		if interp == 'nearest':
			interpflag = cv2.INTER_NEAREST
		elif interp == 'cubic':
			interpflag = cv2.INTER_CUBIC
		elif interp == 'lanczos':
			interpflag = cv2.INTER_LANCZOS4
		else:
			interpflag = cv2.INTER_LINEAR 

		if border == 'constant':
			borderflag = cv2.BORDER_CONSTANT
		else:
			borderflag = cv2.BORDER_REPLICATE


		im = cv2.warpAffine(im, M, (cols, rows), flags = interpflag, borderMode = borderflag)
								
		# Save processed image to HDF5 file (atomic procedure - lock used):
		_write_data(lock, im, i, outfile, outshape, outtype, rescale_factor, logfilename, t2 - t1, t1 - t0)


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
	rescale_factor = 10000.0 # For 16-bit floating point

	# Get the from and to number of files to process:
	int_from = int(argv[0])
	int_to = int(argv[1])
	   
	# Get paths:
	infile = argv[2]
	outfile = argv[3]	
	
	# Params for flat fielding with post flats/darks:
	flat_end = True if argv[4] == "True" else False
	half_half = True if argv[5] == "True" else False
	half_half_line = int(argv[6])

	# Flat fielding method (conventional or dynamic):
	dynamic_ff = True if argv[7] == "True" else False

	# Parameters for rotation:
	rotation = float(argv[8])
	interp = argv[9]
	border = argv[10]
	
	# Nr of threads and log file:
	nr_threads = int(argv[11])
	logfilename = argv[12]		




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
		if "/provenance/detector_output" in f_in:
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
	if ((int_to >= num_proj) or (int_to == -1)):
		int_to = num_proj - 1

	# Prepare the work plan for flat and dark images:
	log = open(logfilename,"a")
	log.write(linesep + "\t--------------")
	log.write(linesep + "\tPreparing the work plan...")				
	log.close()

	# Extract flat and darks:
	skipflat = False
	skipdark = False

	# Following variables make sense only for dynamic flat fielding:
	EFF = -1
	filtEFF = -1
	im_dark = -1
	
	# Following variable makes sense only for conventional flat fielding:
	plan = -1

	if not dynamic_ff:
		plan = extract_flatdark(f_in, flat_end, logfilename)
		if (isscalar(plan['im_flat']) and isscalar(plan['im_flat_after'])):
			skipflat = True
		else:
			skipflat = False		
	else:
		# Dynamic flat fielding:
		if "/tomo" in f_in:				
			if "/flat" in f_in:
				flat_dset = f_in['flat']
				if "/dark" in f_in:
					im_dark = _medianize(f_in['dark'])
				else:										
					skipdark = True
			else:
				skipflat = True # Nothing to do in this case
		else: 
			if "/exchange/data_white" in f_in:
				flat_dset = f_in['/exchange/data_white']
				if "/exchange/data_dark" in f_in:
					im_dark = _medianize(f_in['/exchange/data_dark'])
				else:					
					skipdark = True
			else:
				skipflat = True # Nothing to do in this case
	
		# Prepare plan for dynamic flat fielding with 16 repetitions:
		if not skipflat:	
			EFF, filtEFF = dff_prepare_plan(flat_dset, 16, im_dark)
	
	# Get the corrected outshape (in this case it's easy):
	im = tdf.read_tomo(dset,0).astype(float32)	
	outshape = tdf.get_dset_shape(im.shape[1], im.shape[0], num_proj)			
	
	# Create the output HDF5 file:
	f_out = getHDF5(outfile, 'w')
	#f_out_dset = f_out.create_dataset('exchange/data', outshape, im.dtype)
	f_out_dset = f_out.create_dataset('exchange/data', outshape, float16) 
	f_out_dset.attrs['min'] = str(amin(im[:]))
	f_out_dset.attrs['max'] = str(amax(im[:]))
	f_out_dset.attrs['version'] = '1.0'
	f_out_dset.attrs['axes'] = "y:theta:x"
	f_out_dset.attrs['rescale_factor'] = str(rescale_factor)

	f_out.close()
	f_in.close()
		
	# Log infos:
	log = open(logfilename,"a")
	log.write(linesep + "\tWork plan prepared correctly.")	
	log.write(linesep + "\t--------------")
	log.write(linesep + "\tPerforming pre processing...")			
	log.close()	

	# Run several threads for independent computation without waiting for threads
	# completion:
	for num in range(nr_threads):
		start = (num_proj / nr_threads) * num
		if (num == nr_threads - 1):
			end = num_proj - 1
		else:
			end = (num_proj / nr_threads) * (num + 1) - 1
		Process(target=_process, args=(lock, start, end, infile, outfile, outshape, float16, skipflat, plan, 
				   flat_end, half_half, half_half_line, dynamic_ff, EFF, filtEFF, im_dark, rotation, interp, border, 
				   rescale_factor, logfilename)).start()


	#start = int_from # 0
	#end = int_to # num_sinos - 1
	#_process(lock, start, end, infile, outfile, outshape, float16, skipflat, plan, flat_end, half_half, 
	#			half_half_line, dynamic_ff, EFF, filtEFF, im_dark, rotation, interp, border, rescale_factor, logfilename)

	#255 256 C:\Temp\BrunGeorgos.tdf C:\Temp\BrunGeorgos_corr.tdf 0 0 True True 900 False False 0 rivers:11;0 False 1 C:\Temp\log_00.txt

	
if __name__ == "__main__":
	main(argv[1:])
