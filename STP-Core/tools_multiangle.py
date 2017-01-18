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
# Last modified: Sept, 28th 2016
#

# python:
from sys import argv, exit
from os import remove, sep, linesep, listdir, makedirs
from os.path import exists, dirname, basename, splitext
from numpy import array, finfo, copy, float32, double, amin, amax, tile, concatenate, asarray
from numpy import empty, reshape, log as nplog, arange, squeeze, fromfile, ndarray, where, meshgrid
from time import time
from multiprocessing import Process, Lock

# pystp-specific:
from preprocess.extfov_correction import extfov_correction
from preprocess.flat_fielding import flat_fielding
from preprocess.ring_correction import ring_correction
from preprocess.extract_flatdark import extract_flatdark

from phaseretrieval.tiehom import tiehom, tiehom_plan
from phaseretrieval.phrt   import phrt, phrt_plan

from reconstruct.rec_astra import recon_astra_fbp, recon_astra_iterative
from reconstruct.rec_fista_tv import recon_fista_tv
from reconstruct.rec_mr_fbp import recon_mr_fbp
from reconstruct.rec_gridrec import recon_gridrec

from postprocess.postprocess import postprocess

from utils.padding import upperPowerOfTwo, padImage, padSmoothWidth
from utils.caching import cache2plan, plan2cache

from tifffile import imread, imsave
from h5py import File as getHDF5
import stpio.tdf as tdf


def write_log(lock, fname, logfilename):    	      
	"""To do...

	"""
	lock.acquire()
	try: 
		# Print out execution time:
		log = open(logfilename,"a")
		log.write(linesep + "\t%s reconstructed." % basename(fname))
		log.close()	

	finally:
		lock.release()	

def reconstruct(im, angles, offset, logtransform, param1, circle, scale, pad, method, 
				zerone_mode, dset_min, dset_max, corr_offset, postprocess_required, convert_opt, 
				crop_opt, start, end, outpath, sino_idx, downsc_factor, decim_factor, logfilename, lock, slice_prefix):
	"""Reconstruct a sinogram with FBP algorithm (from ASTRA toolbox).

	Parameters
	----------
	im1 : array_like
		Sinogram image data as numpy array.
	center : float
		Offset of the center of rotation to use for the tomographic 
		reconstruction with respect to the half of sinogram width 
		(default=0, i.e. half width).
	logtransform : boolean
		Apply logarithmic transformation before reconstruction (default=True).
	filter : string
		Filter to apply before the application of the reconstruction algorithm. Filter 
		types are: ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos, triangular, 
		gaussian, barlett-hann, blackman, nuttall, blackman-harris, blackman-nuttall, 
		flat-top, kaiser, parzen.
	circle : boolean
		Create a circle in the reconstructed image and set to zero pixels outside the 
		circle (default=False).	
	
	"""
	# Copy required due to multithreading:
	im_f = im

	# Decimate projections if required:
	#if decim_factor > 1:
	#	im = im[::decim_factor,:]	
	
	# Upscale projections (if required):
	if (abs(scale - 1.0) > finfo(float32).eps):		
		siz_orig1 = im_f.shape[1]		
		im_f = imresize(im_f, (im_f.shape[0], int(round(scale * im_f.shape[1]))), interp='bicubic', mode='F')
		offset = int(offset * scale)	
	
	offset = int(round(offset))

	# Apply transformation for changes in the center of rotation:
	if (offset != 0):
		if (offset >= 0):
			im_f = im_f[:,:-offset]
		
			tmp = im_f[:,0] # Get first column
			tmp = tile(tmp, (offset,1)) # Replicate the first column the right number of times
			im_f = concatenate((tmp.T,im_f), axis=1) # Concatenate tmp before the image
						
		else:
			im_f = im_f[:,abs(offset):] 	
		
			tmp = im_f[:,im_f.shape[1] - 1] # Get last column
			tmp = tile(tmp, (abs(offset),1)) # Replicate the last column the right number of times
			im_f = concatenate((im_f,tmp.T), axis=1) # Concatenate tmp after the image	
	
	# Downscale projections (without pixel averaging):
	#if downsc_factor > 1:
	#	im = im[:,::downsc_factor]			
			
	# Scale image to [0,1] range (if required):
	if (zerone_mode):
		
		#print dset_min
		#print dset_max
		#print numpy.amin(im_f[:])
		#print numpy.amax(im_f[:])
		#im_f = (im_f - dset_min) / (dset_max - dset_min)
		
		# Cheating the whole process:
		im_f = (im_f - numpy.amin(im_f[:])) / (numpy.amax(im_f[:]) - numpy.amin(im_f[:]))
			
	# Apply log transform:
	if (logtransform == True):						
		im_f[im_f <= finfo(float32).eps] = finfo(float32).eps
		im_f = -nplog(im_f + corr_offset)	
	
	# Replicate pad image to double the width:
	if (pad):	

		dim_o = im_f.shape[1]		
		n_pad = im_f.shape[1] + im_f.shape[1] / 2					
		marg  = (n_pad - dim_o) / 2	
	
		# Pad image:
		im_f = padSmoothWidth(im_f, n_pad)		

	# Loop for all the required angles:
	for i in range(int(round(start)), int(round(end)) + 1):      	

		# Save image for next step:
		im = im_f
		
		# Apply projection removal (if required):
		im_f = im_f[0:int(round(i/decim_factor)), :]	
	
		# Perform the actual reconstruction:
		if (method.startswith('FBP')):
			im_f = recon_astra_fbp(im_f, angles, method, param1)	
		elif (method == 'MR-FBP_CUDA'):
			im_f = recon_mr_fbp(im_f, angles)
		elif (method == 'FISTA-TV_CUDA'):
			im_f = recon_fista_tv(im_f, angles, param1, param1)
		elif (method == 'MLEM'):
			im_f = recon_tomopy_iterative(im_f, angles, method, param1)		
		elif (method == 'GRIDREC'):
			[im_f, im_f] = recon_gridrec(im_f, im_f, angles, param1)		
		else:
			im_f = recon_astra_iterative(im_f, angles, method, param1, zerone_mode)	
		
		# Crop:
		if (pad):		
			im_f = im_f[marg:dim_o + marg, marg:dim_o + marg]			

		# Resize (if necessary):
		if (abs(scale - 1.0) > finfo(float32).eps):
			im_f = imresize(im_f, (siz_orig1, siz_orig1), interp='nearest', mode='F')

		# Apply post-processing (if required):
		if postprocess_required:
			im_f = postprocess(im_f, convert_opt, crop_opt)
		else:
			# Create the circle mask for fancy output:
			if (circle == True):
				siz = im_f.shape[1]
				if siz % 2:
					rang = arange(-siz / 2 + 1, siz / 2 + 1)
				else:
					rang = arange(-siz / 2,siz / 2)
				x,y = meshgrid(rang,rang)
				z = x ** 2 + y ** 2
				a = (z < (siz / 2 - int(round(abs(offset)/downsc_factor)) ) ** 2)
				im_f = im_f * a			

		# Write down reconstructed image (file name modified with metadata):		
		fname = outpath + slice_prefix + '_' + str(sino_idx).zfill(4) + '_off=' + str(offset*downsc_factor).zfill(4) + '_proj=' + str(i).zfill(4) + '.tif'
		imsave(fname, im_f)	

		# Restore original image for next step:
		im_f = im

		# Write log (atomic procedure - lock used):
		write_log(lock, fname, logfilename )

		
def process(sino_idx, num_sinos, infile, outpath, preprocessing_required, corr_plan, norm_sx, norm_dx, flat_end, half_half, 
			half_half_line, ext_fov, ext_fov_rot_right, ext_fov_overlap, ext_fov_normalize, ext_fov_average,
			ringrem, phaseretrieval_required, phrtmethod, phrt_param1, phrt_param2, 
			energy, distance, pixsize, phrtpad, approx_win, angles, offset, logtransform, param1, circle, scale, pad, method, 
			zerone_mode, dset_min, dset_max, decim_factor, downsc_factor, corr_offset, postprocess_required, convert_opt, 
			crop_opt, nr_threads, angles_from, angles_to, logfilename, lock, slice_prefix):
	"""To do...

	"""
	slice_nr = sino_idx
	
	# Perform reconstruction (on-the-fly preprocessing and phase retrieval, if required):
	if (phaseretrieval_required):
		
		# In this case a bunch of sinograms is loaded into memory:		

		#
		# Load the temporary data structure reading the input TDF file.
		# To know the right dimension the first sinogram is pre-processed.
		#		

		# Open the TDF file and get the dataset:
		f_in = getHDF5(infile, 'r')
		if "/tomo" in f_in:
			dset = f_in['tomo']
		else: 
			dset = f_in['exchange/data']
		
		# Downscaling and decimation factors considered when determining the approximation window:
		zrange = arange(sino_idx - approx_win*downsc_factor/2, sino_idx + approx_win*downsc_factor/2, downsc_factor)
		zrange = zrange[ (zrange >= 0) ]
		zrange = zrange[ (zrange < num_sinos) ]
		approx_win = zrange.shape[0]
		
		# Approximation window cannot be odd:
		if (approx_win % 2 == 1):
			approx_win = approx_win-1 
			zrange     = zrange[0:approx_win]
		
		# Read one sinogram to get the proper dimensions:
		test_im = tdf.read_sino(dset, zrange[0]*downsc_factor).astype(float32)		
		
		# Apply decimation and downscaling (if required):
		test_im = test_im[::decim_factor, ::downsc_factor]

		# Perform the pre-processing of the first sinogram to get the right dimension:
		if (preprocessing_required):
			test_im = flat_fielding (test_im, zrange[0], corr_plan, flat_end, half_half, 
										half_half_line/decim_factor, norm_sx, norm_dx).astype(float32)	
			if ext_fov:
				test_im = extfov_correction (test_im, ext_fov_rot_right, ext_fov_overlap/downsc_factor, ext_fov_normalize, ext_fov_average).astype(float32)			
			test_im = ring_correction (test_im, ringrem, flat_end, corr_plan['skip_flat_after'], half_half, 
											half_half_line/decim_factor, ext_fov).astype(float32)	
		
		# Now we can allocate memory for the bunch of slices:		
		tmp_im = empty((approx_win, test_im.shape[0], test_im.shape[1]), dtype=float32)
		tmp_im[0,:,:] = test_im

		# Reading all the the sinos from TDF file and close:
		for ct in range(1, approx_win):
			
			test_im = tdf.read_sino(dset, zrange[ct]*downsc_factor).astype(float32)
			test_im = test_im[::decim_factor, ::downsc_factor]
			
			# Perform the pre-processing for each sinogram of the bunch:
			if (preprocessing_required):
				test_im = flat_fielding (test_im, zrange[ct], corr_plan, flat_end, half_half, 
											half_half_line/decim_factor, norm_sx, norm_dx).astype(float32)	
				if ext_fov:
					test_im = extfov_correction (test_im, ext_fov_rot_right, ext_fov_overlap/downsc_factor, ext_fov_normalize, ext_fov_average).astype(float32)			
				test_im = ring_correction (test_im, ringrem, flat_end, corr_plan['skip_flat_after'], half_half, 
											half_half_line/decim_factor, ext_fov).astype(float32)	
			
			tmp_im[ct,:,:] = test_im
	
		f_in.close()

		# Now everything has to refer to a downscaled dataset:
		sino_idx = ((zrange == sino_idx).nonzero())

		#
		# Perform phase retrieval:
		#

		# Prepare the plan:		
		if (phrtmethod == 0):
			# Paganin's:
			phrtplan = tiehom_plan (tmp_im[:,0,:], phrt_param1, phrt_param2, energy, distance, pixsize*downsc_factor, padding=phrtpad)
		else:
			phrtplan = phrt_plan (tmp_im[:,0,:], energy, distance, pixsize*downsc_factor, phrt_param2, phrt_param1, phrtmethod, padding=phrtpad)
		
		# Process each projection (whose height depends on the size of the bunch):
		for ct in range(0, tmp_im.shape[1]):
			if (phrtmethod == 0):
				tmp_im[:,ct,:] = tiehom(tmp_im[:,ct,:], phrtplan).astype(float32)			
			else:
				tmp_im[:,ct,:] = phrt(tmp_im[:,ct,:], phrtplan, phrtmethod).astype(float32)	
		
		# Extract the requested sinogram:
		im = tmp_im[sino_idx[0],:,:].squeeze()	

	else:

		# Read only one sinogram:
		f_in = getHDF5(infile, 'r')
		if "/tomo" in f_in:
			dset = f_in['tomo']
		else: 
			dset = f_in['exchange/data']
		im = tdf.read_sino(dset,sino_idx * downsc_factor).astype(float32)		
		f_in.close()

		# Downscale and decimate the sinogram:
		im = im[::decim_factor,::downsc_factor]
		#sino_idx = sino_idx/downsc_factor	
			
		# Perform the preprocessing of the sinogram (if required):
		if (preprocessing_required):
			im = flat_fielding (im, sino_idx, corr_plan, flat_end, half_half, half_half_line/decim_factor, 
								norm_sx, norm_dx).astype(float32)		
			if ext_fov:
				test_im = extfov_correction (test_im, ext_fov_rot_right, ext_fov_overlap/downsc_factor, ext_fov_normalize, ext_fov_average).astype(float32)			
			im = ring_correction (im, ringrem, flat_end, corr_plan['skip_flat_after'], half_half, 
								half_half_line/decim_factor, ext_fov)

	# Log infos:
	log = open(logfilename,"a")	
	log.write(linesep + "\tPerforming reconstruction with multiple centers of rotation...")			
	log.write(linesep + "\t--------------")		
	log.close()	

	# Split the computation into multiple processes:
	for num in range(nr_threads):
		start = ( (angles_to - angles_from + 1) / nr_threads)*num + angles_from
		if (num == nr_threads - 1):
			end = angles_to
		else:
			end = ( (angles_to - angles_from + 1) / nr_threads)*(num + 1) + angles_from - 1

		Process(target=reconstruct, args=(im, angles, offset/downsc_factor, logtransform, param1, circle, scale, pad, method, 
					   zerone_mode, dset_min, dset_max, corr_offset, postprocess_required, convert_opt, crop_opt, start, end, 
		               outpath, slice_nr, downsc_factor, decim_factor, logfilename, lock, slice_prefix)).start()


		# Actual reconstruction:
		#reconstruct(im, angles, offset/downsc_factor, logtransform, param1, circle, scale, pad, method, 
		#				zerone_mode, dset_min, dset_max, corr_offset, postprocess_required, convert_opt, crop_opt, 
		#				start, end, outpath, slice_nr, downsc_factor, decim_factor, logfilename, lock, slice_prefix)

										


def main(argv):          
	"""To do...


	"""
	lock = Lock()
	skip_flat = False
	skip_flat_after = True	

	# Get the from and to number of files to process:
	sino_idx = int(argv[0])
	   
	# Get paths:
	infile  = argv[1]
	outpath = argv[2]

	# Essential reconstruction parameters::
	angles   = float(argv[3])
	offset   = float(argv[4])
	param1   = argv[5]	
	scale    = int(float(argv[6]))
	
	overpad  = True if argv[7] == "True" else False
	logtrsf  = True if argv[8] == "True" else False
	circle   = True if argv[9] == "True" else False
	
	# Parameters for on-the-fly pre-processing:
	preprocessing_required = True if argv[10] == "True" else False		
	flat_end = True if argv[11] == "True" else False		
	half_half = True if argv[12] == "True" else False
		
	half_half_line = int(argv[13])
		
	ext_fov = True if argv[14] == "True" else False
		
	norm_sx = int(argv[19])
	norm_dx = int(argv[20])	
		
	ext_fov_rot_right = argv[15]
	if ext_fov_rot_right == "True":
		ext_fov_rot_right = True
		if (ext_fov):
			norm_sx = 0
	else:
		ext_fov_rot_right = False
		if (ext_fov):
			norm_dx = 0
		
	ext_fov_overlap = int(argv[16])

	ext_fov_normalize = True if argv[17] == "True" else False
	ext_fov_average = True if argv[18] == "True" else False
		
	skip_ringrem = True if argv[21] == "True" else False
	ringrem = argv[22]
	
	# Extra reconstruction parameters:
	zerone_mode = True if argv[23] == "True" else False		
	corr_offset = float(argv[24])
		
	reconmethod = argv[25]	
	
	decim_factor = int(argv[26])
	downsc_factor = int(argv[27])
	
	# Parameters for postprocessing:
	postprocess_required = True if argv[28] == "True" else False
	convert_opt = argv[29]
	crop_opt = argv[30]

	# Parameters for on-the-fly phase retrieval:
	phaseretrieval_required = True if argv[31] == "True" else False		
	phrtmethod = int(argv[32])
	phrt_param1 = double(argv[33])   # param1( e.g. regParam, or beta)
	phrt_param2 = double(argv[34])   # param2( e.g. thresh or delta)
	energy = double(argv[35])
	distance = double(argv[36])    
	pixsize = double(argv[37]) / 1000.0 # pixsixe from micron to mm:	
	phrtpad = True if argv[38] == "True" else False
	approx_win = int(argv[39])	

	preprocessingplan_fromcache = True if argv[40] == "True" else False
	tmppath    = argv[41]	
	if not tmppath.endswith(sep): tmppath += sep

	nr_threads = int(argv[42])	
	angles_from = float(argv[43])
	angles_to   = float(argv[44])

	slice_prefix = argv[45]
		
	logfilename = argv[46]	

	if not exists(outpath):
		makedirs(outpath)
	
	if not outpath.endswith(sep): outpath += sep	


	# Log info:
	log = open(logfilename,"w")
	log.write(linesep + "\tInput dataset: %s" % (infile))	
	log.write(linesep + "\tOutput path: %s" % (outpath))		
	log.write(linesep + "\t--------------")		
	log.write(linesep + "\tLoading flat and dark images...")	
	log.close()	
			
	# Open the HDF5 file:
	f_in = getHDF5(infile, 'r')
	if "/tomo" in f_in:
		dset = f_in['tomo']	
	else: 
		dset = f_in['exchange/data']
		if "/provenance/detector_output" in f_in:
			prov_dset = f_in['provenance/detector_output']				
	
	dset_min = -1
	dset_max = -1
	if (zerone_mode):
		if ('min' in dset.attrs):
			dset_min = float(dset.attrs['min'])								
		else:
			zerone_mode = False
			
		if ('max' in dset.attrs):
			dset_max = float(dset.attrs['max'])				
		else:
			zerone_mode = False	
		
	num_sinos = tdf.get_nr_sinos(dset) # Pay attention to the downscale factor
	
	if (num_sinos == 0):	
		exit()		

	# Check extrema:
	if (sino_idx >= num_sinos / downsc_factor):
		sino_idx = num_sinos / downsc_factor - 1
	
	# Get correction plan and phase retrieval plan (if required):
	corrplan = 0	
	if (preprocessing_required):		
		# Load flat fielding plan either from cache (if required) or from TDF file and cache it for faster re-use:
		if (preprocessingplan_fromcache):
			try:
				corrplan = cache2plan(infile, tmppath)
			except Exception as e:
				#print "Error(s) when reading from cache"
				corrplan = extract_flatdark(f_in, flat_end, logfilename)
				plan2cache(corrplan, infile, tmppath)
		else:			
			corrplan = extract_flatdark(f_in, flat_end, logfilename)		
			plan2cache(corrplan, infile, tmppath)	

		# Dowscale flat and dark images if necessary:
		if isinstance(corrplan['im_flat'], ndarray):
			corrplan['im_flat'] = corrplan['im_flat'][::downsc_factor,::downsc_factor]		
		if isinstance(corrplan['im_dark'], ndarray):
			corrplan['im_dark'] = corrplan['im_dark'][::downsc_factor,::downsc_factor]	
		if isinstance(corrplan['im_flat_after'], ndarray):
			corrplan['im_flat_after'] = corrplan['im_flat_after'][::downsc_factor,::downsc_factor]	
		if isinstance(corrplan['im_dark_after'], ndarray):
			corrplan['im_dark_after'] = corrplan['im_dark_after'][::downsc_factor,::downsc_factor]			

	f_in.close()	

	# Log infos:
	log = open(logfilename,"a")
	log.write(linesep + "\tPerforming preprocessing...")			
	log.close()			

	# Run computation:	
	process( sino_idx, num_sinos, infile, outpath, preprocessing_required, corrplan, norm_sx, 
			norm_dx, flat_end, half_half, half_half_line, ext_fov, ext_fov_rot_right, ext_fov_overlap, 
			ext_fov_normalize, ext_fov_average, ringrem, 
			phaseretrieval_required, phrtmethod, phrt_param1, phrt_param2, energy, distance, pixsize, phrtpad, 
			approx_win, angles, offset, logtrsf, param1, circle, scale, overpad, reconmethod, zerone_mode, 
			dset_min, dset_max, decim_factor, downsc_factor, corr_offset, postprocess_required, convert_opt, 
			crop_opt, nr_threads, angles_from, angles_to, logfilename, lock, slice_prefix )		

	# Sample:
	# 26 C:\Temp\dataset42.tdf C:\Temp\test_angles 3.1416 -72.0 shepp-logan 1.0 False True True True True False 5 False False 100 0 0 False rivers:11;0 False 0.0 FBP_CUDA 1 4 False - - False 0 1.0 1000.0 22 150 2.2 True 16 False C:\Temp 1 1148 1248 slice C:\Temp\log_angles_00.txt



if __name__ == "__main__":
	main(argv[1:])


