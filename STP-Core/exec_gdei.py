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
# Last modified: April, 5th 2017
#

from sys import argv, exit
from os import remove, sep,  linesep
from os.path import exists
from numpy import float32, amin, amax, isscalar, finfo, empty_like, minimum, maximum, tile, log, mean, array
from time import time
from multiprocessing import Process, Lock

from preprocess.gdei import gdei
from preprocess.extfov_correction import extfov_correction
from preprocess.flat_fielding import flat_fielding
from preprocess.dynamic_flatfielding import dff_prepare_plan, dynamic_flat_fielding
from preprocess.ring_correction import ring_correction
from preprocess.extract_flatdark import extract_flatdark, _medianize

from h5py import File as getHDF5
import stpio.tdf as tdf

from tifffile import imsave

def _shift_vert(im, n):
	out = empty_like(im)
	if n >= 0:
		out[:n,:] = tile(im[0,:],(n,1)) 
		out[n:,:] = im[:-n,:]
	else:
		out[n:,:] = tile(im[-1,:],(-n,1)) 
		out[:n,:] = im[-n:,:]
	return out

def _shift_horiz(im, n):
	out = empty_like(im)
	if n >= 0:
		out[:,:n] = tile(im[:,0], (n,1)).T
		out[:,n:] = im[:,:-n]
	else:
		out[:,n:] = tile(im[:,-1], (-n,1)).T
		out[:,:n] = im[:,-n:]
	return out

def _write_data(im, index, outfile, outshape, outtype):    	      

			
		f_out = getHDF5(outfile, 'a')					 
		f_out_dset = f_out.require_dataset('exchange/data', outshape, outtype, chunks=tdf.get_dset_chunks(outshape[0])) 
		tdf.write_sino(f_out_dset,index,im.astype(outtype))
					
		# Set minimum and maximum:
		if (amin(im[:]) < float(f_out_dset.attrs['min'])):
			f_out_dset.attrs['min'] = str(amin(im[:]))
		if (amax(im[:]) > float(f_out_dset.attrs['max'])):
			f_out_dset.attrs['max'] = str(amax(im[:]))		
		f_out.close()			
		

def _process(lock, int_from, int_to, num_sinos, infile_1, infile_2, infile_3, outfile_abs, outfile_ref, outfile_sca, 
			 r1, r2, r3, d1, d2, d3, dd1, dd2, dd3, 
			 shiftVert_1, shiftHoriz_1, shiftVert_2, shiftHoriz_2, shiftVert_3, shiftHoriz_3, 
			 outshape, outtype, skipflat_1, skipflat_2, skipflat_3, plan_1, plan_2, plan_3, norm_sx, norm_dx, flat_end, 
			 half_half, half_half_line, ext_fov, ext_fov_rot_right, ext_fov_overlap, ext_fov_normalize, ext_fov_average,
			 ringrem, dynamic_ff, EFF_1, EFF_2, EFF_3, filtEFF_1, filtEFF_2, filtEFF_3, im_dark_1, im_dark_2, im_dark_3, 
			 logfilename):

	# Process the required subset of images:
	for i in range(int_from, int_to + 1):                 
				
		# Read input image for top, left and right:
		t0 = time()
		f_in = getHDF5(infile_1, 'r')
		if "/tomo" in f_in:
			dset = f_in['tomo']
			flat_avg1 = float(f_in['flat'].attrs['avg'])
			dark_avg1 = float(f_in['dark'].attrs['avg'])
		else: 
			dset = f_in['exchange/data']
			flat_avg1 = float(f_in['exchange/data_white'].attrs['avg'])
			dark_avg1 = float(f_in['exchange/data_dark'].attrs['avg'])
		# Processing in the sinogram domain so a vertical shift of the
		# projection requires loading a different sinogram:
		idx1 = min(max(0,i - shiftVert_1),num_sinos-1)
		im_1 = tdf.read_sino(dset, idx1).astype(float32)		
		f_in.close()

		f_in = getHDF5(infile_2, 'r')
		if "/tomo" in f_in:
			dset = f_in['tomo']
			flat_avg2 = float(f_in['flat'].attrs['avg'])
			dark_avg2 = float(f_in['dark'].attrs['avg'])
		else: 
			dset = f_in['exchange/data']
			flat_avg2 = float(f_in['exchange/data_white'].attrs['avg'])
			dark_avg2 = float(f_in['exchange/data_dark'].attrs['avg'])
		# Processing in the sinogram domain so a vertical shift of the
		# projection requires loading a different sinogram:
		idx2 = min(max(0,i - shiftVert_2),num_sinos-1)
		im_2 = tdf.read_sino(dset, idx2).astype(float32)		
		f_in.close()

		f_in = getHDF5(infile_3, 'r')
		if "/tomo" in f_in:
			dset = f_in['tomo']
			flat_avg3 = float(f_in['flat'].attrs['avg'])
			dark_avg3 = float(f_in['dark'].attrs['avg'])
		else: 
			dset = f_in['exchange/data']
			flat_avg3 = float(f_in['exchange/data_white'].attrs['avg'])
			dark_avg3 = float(f_in['exchange/data_dark'].attrs['avg'])
		# Processing in the sinogram domain so a vertical shift of the projection
		# requires loading a different sinogram:
		idx3 = min(max(0,i - shiftVert_3),num_sinos-1)
		im_3 = tdf.read_sino(dset, idx3).astype(float32)		
		f_in.close()
		t1 = time() 	

		# Perform pre-processing (flat fielding, extended FOV, ring removal):
		if not skipflat_1:
			if dynamic_ff:
				# Dynamic flat fielding with downsampling = 2:
				im_1 = dynamic_flat_fielding(im_1, idx1, EFF_1, filtEFF_1, 2, im_dark_1, norm_sx, norm_dx)				
			else:
				im_1 = flat_fielding(im_1, idx1, plan_1, flat_end, half_half, half_half_line, norm_sx, norm_dx)		
		if ext_fov:
			im_1 = extfov_correction(im_1, ext_fov_rot_right, ext_fov_overlap, ext_fov_normalize, ext_fov_average)
		
		if not skipflat_1 and not dynamic_ff:
			im_1 = ring_correction(im_1, ringrem, flat_end, plan_1['skip_flat_after'], half_half, half_half_line, ext_fov)
		else:
			im_1 = ring_correction(im_1, ringrem, False, False, half_half, half_half_line, ext_fov)

		# Perform pre-processing (flat fielding, extended FOV, ring removal):
		if not skipflat_2:
			if dynamic_ff:
				# Dynamic flat fielding with downsampling = 2:
				im_2 = dynamic_flat_fielding(im_2, idx2, EFF_2, filtEFF_2, 2, im_dark_2, norm_sx, norm_dx)
			else:
				im_2 = flat_fielding(im_2, idx2, plan_2, flat_end, half_half, half_half_line, norm_sx, norm_dx)			
		if ext_fov:
			im_2 = extfov_correction(im_2, ext_fov_rot_right, ext_fov_overlap, ext_fov_normalize, ext_fov_average)
		
		if not skipflat_2 and not dynamic_ff:
			im_2 = ring_correction(im_2, ringrem, flat_end, plan_2['skip_flat_after'], half_half, half_half_line, ext_fov)
		else:
			im_2 = ring_correction(im_2, ringrem, False, False, half_half, half_half_line, ext_fov)

		# Perform pre-processing (flat fielding, extended FOV, ring removal):
		if not skipflat_3:
			if dynamic_ff:
				# Dynamic flat fielding with downsampling = 2:
				im_3 = dynamic_flat_fielding(im_3, idx3, EFF_3, filtEFF_3, 2, im_dark_3, norm_sx, norm_dx)
			else:
				im_3 = flat_fielding(im_3, idx3, plan_3, flat_end, half_half, half_half_line, norm_sx, norm_dx)			
		if ext_fov:
			im_3 = extfov_correction(im_3, ext_fov_rot_right, ext_fov_overlap, ext_fov_normalize, ext_fov_average)
		
		if not skipflat_3 and not dynamic_ff:
			im_3 = ring_correction(im_3, ringrem, flat_end, plan_3['skip_flat_after'], half_half, half_half_line, ext_fov)
		else:
			im_3 = ring_correction(im_3, ringrem, False, False, half_half, half_half_line, ext_fov)

		t2 = time() 	


		# Processing in the sinogram domain so a vertical shift of the
		# projection requires loading a different sinogram:
		# Only horizontal shift can be considered at a sinogram level:
		if (shiftHoriz_1 != 0):
			im_1 = _shift_horiz(im_1, shiftHoriz_1)	

		if (shiftHoriz_2 != 0):
			im_2 = _shift_horiz(im_2, shiftHoriz_2)

		if (shiftHoriz_3 != 0):
			im_3 = _shift_horiz(im_3, shiftHoriz_3)

		# Re-normalize with average of the flat-field images:
		max_val = amax(array([mean(flat_avg1 - dark_avg1), mean(flat_avg2 - dark_avg2), mean(flat_avg3 - dark_avg3)]))

		im_1 = im_1 * (flat_avg1 - dark_avg1) / max_val
		im_2 = im_2 * (flat_avg2 - dark_avg2) / max_val
		im_3 = im_3 * (flat_avg3 - dark_avg3) / max_val

		# Apply GDEI:
		(im_abs, im_ref, im_sca) = gdei(im_1, im_2, im_3, r1, r2, r3, d1, d2, d3, dd1, dd2, dd3)

		# Save processed image to HDF5 file (atomic procedure - lock used):		
		lock.acquire()
		try:        
			t3 = time() 
			_write_data(im_abs, i, outfile_abs, outshape, outtype)
			_write_data(im_ref, i, outfile_ref, outshape, outtype)
			_write_data(im_sca, i, outfile_sca, outshape, outtype)
			t4 = time() 

			# Print out execution time:
			log = open(logfilename,"a")
			log.write(linesep + "\tsino_%s processed (CPU: %0.3f sec - I/O: %0.3f sec)." % (str(i).zfill(4), t2 - t1, (t1 - t0) + (t4 - t3)))
			log.close()	

		finally:
			lock.release()	


def main(argv):          
	"""To do...

	Usage
	-----
	

	Parameters
	---------
		   
	Example
	--------------------------    

	"""
	lock = Lock()

	# Get the from and to number of files to process:
	int_from = int(argv[0])
	int_to = int(argv[1])
	   
	# Get paths:
	infile_1 = argv[2]
	infile_2 = argv[3]
	infile_3 = argv[4]

	outfile_abs = argv[5]
	outfile_ref = argv[6]
	outfile_sca = argv[7]
	
	# Normalization parameters:
	norm_sx = int(argv[8])
	norm_dx = int(argv[9])
	
	# Params for flat fielding with post flats/darks:
	flat_end = True if argv[10] == "True" else False
	half_half = True if argv[11] == "True" else False
	half_half_line = int(argv[12])
		
	# Params for extended FOV:
	ext_fov = True if argv[13] == "True" else False
	ext_fov_rot_right = argv[14]
	if ext_fov_rot_right == "True":
		ext_fov_rot_right = True
		if (ext_fov):
			norm_sx = 0
	else:
		ext_fov_rot_right = False
		if (ext_fov):
			norm_dx = 0		
	ext_fov_overlap = int(argv[15])

	ext_fov_normalize = True if argv[16] == "True" else False
	ext_fov_average = True if argv[17] == "True" else False
		
	# Method and parameters coded into a string:
	ringrem = argv[18]	

	# Flat fielding method (conventional or dynamic):
	dynamic_ff = True if argv[19] == "True" else False
	
	# Shift parameters:
	shiftVert_1 = int(argv[20])
	shiftHoriz_1 = int(argv[21])
	shiftVert_2 = int(argv[22])
	shiftHoriz_2 = int(argv[23])
	shiftVert_3 = int(argv[24])
	shiftHoriz_3 = int(argv[25])

	# DEI coefficients:
	r1 = float(argv[26])
	r2 = float(argv[27])
	r3 = float(argv[28])
	d1 = float(argv[29])
	d2 = float(argv[30])
	d3 = float(argv[31])
	dd1 = float(argv[32])
	dd2 = float(argv[33])
	dd3 = float(argv[34])

	

	# Nr of threads and log file:
	nr_threads = int(argv[35])
	logfilename = argv[36]		


	# Log input parameters:
	log = open(logfilename,"w")
	log.write(linesep + "\tInput TDF file #1: %s" % (infile_1))	
	log.write(linesep + "\tInput TDF file #2: %s" % (infile_2))	
	log.write(linesep + "\tInput TDF file #3: %s" % (infile_3))	
	log.write(linesep + "\tOutput TDF file for Absorption: %s" % (outfile_abs))		
	log.write(linesep + "\tOutput TDF file for Refraction: %s" % (outfile_ref))		
	log.write(linesep + "\tOutput TDF file for Scattering: %s" % (outfile_sca))		
	log.write(linesep + "\t--------------")	
	log.write(linesep + "\tOpening input dataset...")	
	log.close()
	
	# Remove a previous copy of output:
	#if exists(outfile):
	#	remove(outfile)
	
	# Open the HDF5 files:
	f_in_1 = getHDF5(infile_1, 'r')
	f_in_2 = getHDF5(infile_2, 'r')
	f_in_3 = getHDF5(infile_3, 'r')


	if "/tomo" in f_in_1:
		dset_1 = f_in_1['tomo']

		tomoprefix_1 = 'tomo'
		flatprefix_1 = 'flat'
		darkprefix_1 = 'dark'
	else: 
		dset_1 = f_in_1['exchange/data']
		if "/provenance/detector_output" in f_in_1:
			prov_dset_1 = f_in_1['provenance/detector_output']		
	
			tomoprefix_1 = prov_dset_1.attrs['tomo_prefix']
			flatprefix_1 = prov_dset_1.attrs['flat_prefix']
			darkprefix_1 = prov_dset_1.attrs['dark_prefix']

	if "/tomo" in f_in_2:
		dset_2 = f_in_2['tomo']

		tomoprefix_2 = 'tomo'
		flatprefix_2 = 'flat'
		darkprefix_2 = 'dark'
	else: 
		dset_2 = f_in_2['exchange/data']
		if "/provenance/detector_output" in f_in_2:
			prov_dset_2 = f_in_2['provenance/detector_output']		
	
			tomoprefix_2 = prov_dset_2.attrs['tomo_prefix']
			flatprefix_2 = prov_dset_2.attrs['flat_prefix']
			darkprefix_2 = prov_dset_2.attrs['dark_prefix']

	if "/tomo" in f_in_3:
		dset_3 = f_in_3['tomo']

		tomoprefix_3 = 'tomo'
		flatprefix_3 = 'flat'
		darkprefix_3 = 'dark'
	else: 
		dset_3 = f_in_3['exchange/data']
		if "/provenance/detector_output" in f_in_3:
			prov_dset_3 = f_in_1['provenance/detector_output']		
	
			tomoprefix_3 = prov_dset_3.attrs['tomo_prefix']
			flatprefix_3 = prov_dset_3.attrs['flat_prefix']
			darkprefix_3 = prov_dset_3.attrs['dark_prefix']
	
	# Assuming that what works for the dataset #1 works for the other two:
	num_proj = tdf.get_nr_projs(dset_1)
	num_sinos = tdf.get_nr_sinos(dset_1)
	
	if (num_sinos == 0):
		log = open(logfilename,"a")
		log.write(linesep + "\tNo projections found. Process will end.")	
		log.close()			
		exit()		

	# Check extrema (int_to == -1 means all files):
	if ((int_to >= num_sinos) or (int_to == -1)):
		int_to = num_sinos - 1

	# Prepare the work plan for flat and dark images:
	log = open(logfilename,"a")
	log.write(linesep + "\t--------------")
	log.write(linesep + "\tPreparing the work plan...")				
	log.close()

	# Extract flat and darks:
	skipflat_1 = False
	skipdark_1 = False
	skipflat_2 = False
	skipdark_2 = False
	skipflat_3 = False
	skipdark_3 = False

	# Following variables make sense only for dynamic flat fielding:
	EFF_1 = -1
	filtEFF_1 = -1
	im_dark_1 = -1

	EFF_2 = -1
	filtEFF_2 = -1
	im_dark_2 = -1

	EFF_3 = -1
	filtEFF_3 = -1
	im_dark_3 = -1
	
	# Following variable makes sense only for conventional flat fielding:
	plan_1 = -1
	plan_2 = -1
	plan_3 = -1

	if not dynamic_ff:
		plan_1 = extract_flatdark(f_in_1, flat_end, logfilename)
		if (isscalar(plan_1['im_flat']) and isscalar(plan_1['im_flat_after'])):
			skipflat_1 = True
		else:
			skipflat_1 = False	
			
		plan_2 = extract_flatdark(f_in_2, flat_end, logfilename)
		if (isscalar(plan_2['im_flat']) and isscalar(plan_2['im_flat_after'])):
			skipflat_2 = True
		else:
			skipflat_2 = False
			
		plan_3 = extract_flatdark(f_in_3, flat_end, logfilename)
		if (isscalar(plan_3['im_flat']) and isscalar(plan_3['im_flat_after'])):
			skipflat_3 = True
		else:
			skipflat_3 = False	

	else:
		# Dynamic flat fielding:
		if "/tomo" in f_in_1:				
			if "/flat" in f_in_1:
				flat_dset_1 = f_in_1['flat']
				if "/dark" in f_in_1:
					im_dark_1 = _medianize(f_in_1['dark'])
				else:										
					skipdark_1 = True
			else:
				skipflat_1 = True # Nothing to do in this case
		else: 
			if "/exchange/data_white" in f_in_1:
				flat_dset_1 = f_in_1['/exchange/data_white']
				if "/exchange/data_dark" in f_in_1:
					im_dark_1 = _medianize(f_in_1['/exchange/data_dark'])
				else:					
					skipdark_1 = True
			else:
				skipflat_1 = True # Nothing to do in this case
	
		# Prepare plan for dynamic flat fielding with 16 repetitions:
		if not skipflat_1:	
			EFF_1, filtEFF_1 = dff_prepare_plan(flat_dset_1, 16, im_dark_1)

		# Dynamic flat fielding:
		if "/tomo" in f_in_2:				
			if "/flat" in f_in_2:
				flat_dset_2 = f_in_2['flat']
				if "/dark" in f_in_2:
					im_dark_2 = _medianize(f_in_2['dark'])
				else:										
					skipdark_2 = True
			else:
				skipflat_2 = True # Nothing to do in this case
		else: 
			if "/exchange/data_white" in f_in_2:
				flat_dset_2 = f_in_2['/exchange/data_white']
				if "/exchange/data_dark" in f_in_2:
					im_dark_2 = _medianize(f_in_2['/exchange/data_dark'])
				else:					
					skipdark_2 = True
			else:
				skipflat_2 = True # Nothing to do in this case
	
		# Prepare plan for dynamic flat fielding with 16 repetitions:
		if not skipflat_2:	
			EFF_2, filtEFF_2 = dff_prepare_plan(flat_dset_2, 16, im_dark_2)

		# Dynamic flat fielding:
		if "/tomo" in f_in_3:				
			if "/flat" in f_in_3:
				flat_dset_3 = f_in_3['flat']
				if "/dark" in f_in_3:
					im_dark_3 = _medianize(f_in_3['dark'])
				else:										
					skipdark_3 = True
			else:
				skipflat_3 = True # Nothing to do in this case
		else: 
			if "/exchange/data_white" in f_in_3:
				flat_dset_3 = f_in_3['/exchange/data_white']
				if "/exchange/data_dark" in f_in_3:
					im_dark_3 = _medianize(f_in_3['/exchange/data_dark'])
				else:					
					skipdark_3 = True
			else:
				skipflat_3 = True # Nothing to do in this case
	
		# Prepare plan for dynamic flat fielding with 16 repetitions:
		if not skipflat_3:	
			EFF_3, filtEFF_3 = dff_prepare_plan(flat_dset_3, 16, im_dark_3)
	
	# Outfile shape can be determined only after first processing in ext FOV mode:
	if (ext_fov):

		# Read input sino:
		idx = num_sinos / 2
		im = tdf.read_sino(dset_1,idx).astype(float32)				
		im = extfov_correction(im, ext_fov_rot_right, ext_fov_overlap, ext_fov_normalize, ext_fov_average)
		
		# Get the corrected outshape:
		outshape = tdf.get_dset_shape(im.shape[1], num_sinos, im.shape[0])		

	else:
		# Get the corrected outshape (in this case it's easy):
		im = tdf.read_tomo(dset_1,0).astype(float32)	
		outshape = tdf.get_dset_shape(im.shape[1], im.shape[0], num_proj)			
		
	f_in_1.close()
	f_in_2.close()
	f_in_3.close()

	# Create the output HDF5 files:
	f_out_abs = getHDF5(outfile_abs, 'w')	
	f_out_dset_abs = f_out_abs.create_dataset('exchange/data', outshape, float32) 
	f_out_dset_abs.attrs['min'] = str(finfo(float32).max)
	f_out_dset_abs.attrs['max'] = str(finfo(float32).min)
	f_out_dset_abs.attrs['version'] = '1.0'
	f_out_dset_abs.attrs['axes'] = "y:theta:x"
	f_out_abs.close()

	f_out_ref = getHDF5(outfile_ref, 'w')
	f_out_dset_ref = f_out_ref.create_dataset('exchange/data', outshape, float32) 
	f_out_dset_ref.attrs['min'] = str(finfo(float32).max)
	f_out_dset_ref.attrs['max'] = str(finfo(float32).min)
	f_out_dset_ref.attrs['version'] = '1.0'
	f_out_dset_ref.attrs['axes'] = "y:theta:x"
	f_out_ref.close()

	f_out_sca = getHDF5(outfile_sca, 'w')
	f_out_dset_sca = f_out_sca.create_dataset('exchange/data', outshape, float32) 
	f_out_dset_sca.attrs['min'] = str(finfo(float32).max)
	f_out_dset_sca.attrs['max'] = str(finfo(float32).min)
	f_out_dset_sca.attrs['version'] = '1.0'
	f_out_dset_sca.attrs['axes'] = "y:theta:x"
	f_out_sca.close()
	
		
	# Log infos:
	log = open(logfilename,"a")
	log.write(linesep + "\tWork plan prepared correctly.")	
	log.write(linesep + "\t--------------")
	log.write(linesep + "\tPerforming GDEI...")			
	log.close()	

	# Run several threads for independent computation without waiting for threads
	# completion:
	for num in range(nr_threads):
		start = (num_sinos / nr_threads) * num
		if (num == nr_threads - 1):
			end = num_sinos - 1
		else:
			end = (num_sinos / nr_threads) * (num + 1) - 1
		Process(target=_process, args=(lock, start, end, num_sinos, infile_1, infile_2, infile_3, outfile_abs, outfile_ref, outfile_sca,
			  r1, r2, r3, d1, d2, d3, dd1, dd2, dd3, 
			  shiftVert_1, shiftHoriz_1, shiftVert_2, shiftHoriz_2, shiftVert_3, shiftHoriz_3, 
			  outshape, float32, skipflat_1, skipflat_2, skipflat_3, plan_1, plan_2, plan_3, norm_sx, norm_dx, flat_end, 
			  half_half, half_half_line, ext_fov, ext_fov_rot_right, ext_fov_overlap, ext_fov_normalize, ext_fov_average,
			  ringrem, dynamic_ff, EFF_1, EFF_2, EFF_3, filtEFF_1, filtEFF_2, filtEFF_3, im_dark_1, im_dark_2, im_dark_3, 
			  logfilename)).start()


	#start = int_from # 0
	#end = int_to # num_sinos - 1
	#_process(lock, start, end, num_sinos, infile_1, infile_2, infile_3, outfile_abs, outfile_ref, outfile_sca,
	#		  r1, r2, r3, d1, d2, d3, dd1, dd2, dd3, 
	#		  shiftVert_1, shiftHoriz_1, shiftVert_2, shiftHoriz_2, shiftVert_3, shiftHoriz_3, 
	#		  outshape, float32, skipflat_1, skipflat_2, skipflat_3, plan_1, plan_2, plan_3, norm_sx, norm_dx, flat_end, 
	#		  half_half, half_half_line, ext_fov, ext_fov_rot_right, ext_fov_overlap, ext_fov_normalize, ext_fov_average,
	#		  ringrem, dynamic_ff, EFF_1, EFF_2, EFF_3, filtEFF_1, filtEFF_2, filtEFF_3, im_dark_1, im_dark_2, im_dark_3, 
	#		  logfilename)


	#255 256 C:\Temp\BrunGeorgos.tdf C:\Temp\BrunGeorgos_corr.tdf 0 0 True True
	#900 False False 0 rivers:11;0 False 1 C:\Temp\log_00.txt
if __name__ == "__main__":
	main(argv[1:])
