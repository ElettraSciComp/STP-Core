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
from os import remove, sep, linesep, listdir
from os.path import exists, dirname, basename, splitext
from numpy import array, finfo, copy, float32, double, amin, amax, tile, concatenate, asarray, isscalar
from numpy import empty, reshape, log as nplog, arange, squeeze, fromfile, ndarray, where, meshgrid
from time import time
from multiprocessing import Process, Array

# pystp-specific:
from preprocess.extfov_correction import extfov_correction
from preprocess.flat_fielding import flat_fielding
from preprocess.dynamic_flatfielding import dff_prepare_plan, dynamic_flat_fielding
from preprocess.ring_correction import ring_correction
from preprocess.extract_flatdark import extract_flatdark, _medianize

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
import io.tdf as tdf

from preprocess.ringremoval.munchetal import munchetal
from preprocess.ringremoval.boinhaibel import boinhaibel
from preprocess.ringremoval.oimoen import oimoen
from preprocess.ringremoval.raven import raven
from preprocess.ringremoval.rivers import rivers


def reconstruct(im, angles, offset, logtransform, recpar, circle, scale, pad, method, 
				zerone_mode, dset_min, dset_max, corr_offset):
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
	offset = int(round(offset))

	# Upscale projections (if required):
	if (abs(scale - 1.0) > finfo(float32).eps):		
		siz_orig1 = im.shape[1]		
		im = imresize(im, (im.shape[0], int(round(scale * im.shape[1]))), interp='bicubic', mode='F')
		offset = int(offset * scale)		
			
	# Apply transformation for changes in the center of rotation:
	if (offset != 0):
		if (offset >= 0):
			im = im[:,:-offset]
			
			tmp = im[:,0] # Get first column
			tmp = tile(tmp, (offset,1)) # Replicate the first column the right number of times
			im = concatenate((tmp.T,im), axis=1) # Concatenate tmp before the image
						
		else:
			im = im[:,abs(offset):] 	
			
			tmp = im[:,im.shape[1] - 1] # Get last column
			tmp = tile(tmp, (abs(offset),1)) # Replicate the last column the right number of times
			im = concatenate((im,tmp.T), axis=1) # Concatenate tmp after the image	
			
	# Scale image to [0,1] range (if required):
	if (zerone_mode):
		
		#print dset_min
		#print dset_max
		#print numpy.amin(im_f[:])
		#print numpy.amax(im_f[:])
		#im_f = (im_f - dset_min) / (dset_max - dset_min)
		
		# Cheating the whole process:
		im = (im - numpy.amin(im[:])) / (numpy.amax(im[:]) - numpy.amin(im[:]))
			
	# Apply log transform:
	if (logtransform == True):						
		im[im <= finfo(float32).eps] = finfo(float32).eps
		im = -nplog(im + corr_offset)	
	
	# Replicate pad image to double the width:
	if (pad):	

		dim_o = im.shape[1]		
		n_pad = im.shape[1] + im.shape[1] / 2					
		marg  = (n_pad - dim_o) / 2	
	
		# Pad image:
		im = padSmoothWidth(im, n_pad)		
	
	# Perform the actual reconstruction:
	if (method.startswith('FBP')):
		im = recon_astra_fbp(im, angles, method, recpar)	
	elif (method == 'MR-FBP_CUDA'):
		im = recon_mr_fbp(im, angles)
	elif (method == 'FISTA-TV_CUDA'):
		im = recon_fista_tv(im, angles, recpar, recpar)
	elif (method == 'GRIDREC'):
		[im, im] = recon_gridrec(im, im, angles, recpar)	
	else:
		im = recon_astra_iterative(im, angles, method, recpar, zerone_mode)	

		
	# Crop:
	if (pad):		
		im = im[marg:dim_o + marg, marg:dim_o + marg]			

	# Resize (if necessary):
	if (abs(scale - 1.0) > finfo(float32).eps):
		im = imresize(im, (siz_orig1, siz_orig1), interp='nearest', mode='F')

	# Return output:
	return im.astype(float32)


#def _testwritedownsino(tmp_im):

#	for ct in range(0, tmp_im.shape[0]):
#		a = tmp_im[ct,:,:].squeeze()			
#		fname = 'C:\\Temp\\StupidFolder\\sino_' + str(ct).zfill(4) + '.tif'
#		imsave(fname, a.astype(float32))

#def _testwritedownproj(tmp_im):

#	for ct in range(0, tmp_im.shape[1]):
#		a = tmp_im[:,ct,:].squeeze()			
#		fname = 'C:\\Temp\\StupidFolder\\proj_' + str(ct).zfill(4) + '.tif'
#		imsave(fname, a.astype(float32))
		
def process(sino_idx, num_sinos, infile, outfile, preprocessing_required, corr_plan, skipflat, norm_sx, norm_dx, flat_end, half_half, 
			half_half_line, ext_fov, ext_fov_rot_right, ext_fov_overlap, ringrem, phaseretrieval_required, phrtmethod, phrt_param1,
			phrt_param2, energy, distance, pixsize, phrtpad, approx_win, angles, angles_projfrom, angles_projto,
			offset, logtransform, recpar, circle, scale, pad, method, 
			zerone_mode, dset_min, dset_max, decim_factor, downsc_factor, corr_offset, postprocess_required, convert_opt, 
			crop_opt, dynamic_ff, EFF, filtEFF, im_dark, nr_threads, logfilename):
	"""To do...

	"""
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
		test_im = tdf.read_sino(dset, zrange[0]).astype(float32)	

		# Apply projection removal (if required):
		test_im = test_im[angles_projfrom:angles_projto, :]

		# Apply decimation and downscaling (if required):	
		test_im = test_im[::decim_factor, ::downsc_factor]

		# Perform the pre-processing of the first sinogram to get the right dimension:
		if (preprocessing_required):
			if not skipflat:			
				if dynamic_ff:
					# Dynamic flat fielding with downsampling = 2:
					test_im = dynamic_flat_fielding(test_im, zrange[0]/downsc_factor, EFF, filtEFF, 2, im_dark, norm_sx, norm_dx)
				else:
					test_im = flat_fielding(test_im, zrange[0]/downsc_factor, corr_plan, flat_end, half_half, 
											half_half_line/decim_factor, norm_sx, norm_dx).astype(float32)
			test_im = extfov_correction(test_im, ext_fov, ext_fov_rot_right, ext_fov_overlap/downsc_factor).astype(float32)			
			if not skipflat and not dynamic_ff:
				test_im = ring_correction(test_im, ringrem, flat_end, corr_plan['skip_flat_after'], half_half, 
											half_half_line/decim_factor, ext_fov).astype(float32)	
			else:
				test_im = ring_correction(test_im, ringrem, False, False, half_half, 
											half_half_line/decim_factor, ext_fov).astype(float32)	
		
		# Now we can allocate memory for the bunch of slices:		
		tmp_im = empty((approx_win, test_im.shape[0], test_im.shape[1]), dtype=float32)
		tmp_im[0,:,:] = test_im

		# Reading all the the sinos from TDF file and close:
		for ct in range(1, approx_win):

			# Read the sinogram:
			test_im = tdf.read_sino(dset, zrange[ct]).astype(float32)

			# Apply projection removal (if required):
			test_im = test_im[angles_projfrom:angles_projto, :]

			# Apply decimation and downscaling (if required):	
			test_im = test_im[::decim_factor, ::downsc_factor]
			
			# Perform the pre-processing for each sinogram of the bunch:
			if (preprocessing_required):
				if not skipflat:
					if dynamic_ff:
						# Dynamic flat fielding with downsampling = 2:
						test_im = dynamic_flat_fielding(test_im, zrange[ct]/downsc_factor, EFF, filtEFF, 2, im_dark, norm_sx, norm_dx)
					else:
						test_im = flat_fielding (test_im, zrange[ct]/downsc_factor, corr_plan, flat_end, half_half, 
											half_half_line/decim_factor, norm_sx, norm_dx).astype(float32)	
				test_im = extfov_correction (test_im, ext_fov, ext_fov_rot_right, ext_fov_overlap/downsc_factor).astype(float32)
				if not skipflat and not dynamic_ff:
					test_im = ring_correction (test_im, ringrem, flat_end, corr_plan['skip_flat_after'], half_half, 
											half_half_line/decim_factor, ext_fov).astype(float32)	
				else:
					test_im = ring_correction (test_im, ringrem, False, False, half_half, 
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
			phrtplan = tiehom_plan (tmp_im[:,0,:], phrt_param1, phrt_param2, energy, distance, pixsize*downsc_factor, phrtpad)
		else:
			phrtplan = phrt_plan (tmp_im[:,0,:], energy, distance, pixsize*downsc_factor, phrt_param2, phrt_param1, phrtmethod, phrtpad)
			#phrtplan = prepare_plan (tmp_im[:,0,:], beta, delta, energy, distance, pixsize*downsc_factor, padding=phrtpad)
		
		# Process each projection (whose height depends on the size of the bunch):
		for ct in range(0, tmp_im.shape[1]):
			#tmp_im[:,ct,:] = phase_retrieval(tmp_im[:,ct,:], phrtplan).astype(float32)
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
		im = tdf.read_sino(dset,sino_idx).astype(float32)		
		f_in.close()

		# Apply projection removal (if required):
		im = im[angles_projfrom:angles_projto, :]

		# Apply decimation and downscaling (if required):	
		im = im[::decim_factor,::downsc_factor]
		sino_idx = sino_idx/downsc_factor	
			
		# Perform the preprocessing of the sinogram (if required):
		if (preprocessing_required):
			if not skipflat:
				if dynamic_ff:
					# Dynamic flat fielding with downsampling = 2:
					im = dynamic_flat_fielding(im, sino_idx, EFF, filtEFF, 2, im_dark, norm_sx, norm_dx)
				else:
					im = flat_fielding (im, sino_idx, corr_plan, flat_end, half_half, half_half_line/decim_factor, 
								norm_sx, norm_dx).astype(float32)		
			im = extfov_correction (im, ext_fov, ext_fov_rot_right, ext_fov_overlap)
			if not skipflat and not dynamic_ff:
				im = ring_correction (im, ringrem, flat_end, corr_plan['skip_flat_after'], half_half, 
								half_half_line/decim_factor, ext_fov)
			else:
				im = ring_correction (im, ringrem, False, False, half_half, 
								half_half_line/decim_factor, ext_fov)


	# Additional ring removal before reconstruction:
	#im = boinhaibel(im, '11;') 
	#im = munchetal(im, '5;1.8')  
	#im = rivers(im, '13;')   
	#im = raven(im, '11;0.8')
	#im = oimoen(im, '51;51')

	# Actual reconstruction:
	im = reconstruct(im, angles, offset/downsc_factor, logtransform, recpar, circle, scale, pad, method, 
					zerone_mode, dset_min, dset_max, corr_offset).astype(float32)	

	# Apply post-processing (if required):
	if postprocess_required:
		im = postprocess(im, convert_opt, crop_opt)
	else:
		# Create the circle mask for fancy output:
		if (circle == True):
			siz = im.shape[1]
			if siz % 2:
				rang = arange(-siz / 2 + 1, siz / 2 + 1)
			else:
				rang = arange(-siz / 2,siz / 2)
			x,y = meshgrid(rang,rang)
			z = x ** 2 + y ** 2
			a = (z < (siz / 2 - int(round(abs(offset)/downsc_factor)) ) ** 2)
			im = im * a			

    # Write down reconstructed preview file (file name modified with metadata):		
	im = im.astype(float32)
	outfile = outfile + '_' + str(im.shape[1]) + 'x' + str(im.shape[0]) + '_' + str( amin(im)) + '$' + str( amax(im) )	
	im.tofile(outfile)	
								
	#print "With %d thread(s): [%0.3f sec, %0.3f sec, %0.3f sec]." % (nr_threads, t1-t0, t2-t1, t3-t2)	


def main(argv):          
	"""To do...

	Usage
	-----
	

	Parameters
	---------
		   
	Example
	--------------------------


	"""
	# Get the from and to number of files to process:
	sino_idx = int(argv[0])
	   
	# Get paths:
	infile = argv[1]
	outfile = argv[2]

	# Essential reconstruction parameters:
	angles = float(argv[3])	
	offset = float(argv[4])
	recpar = argv[5]	
	scale  = int(float(argv[6]))
	
	overpad = True if argv[7] == "True" else False
	logtrsf = True if argv[8] == "True" else False
	circle = True if argv[9] == "True" else False
	
	# Parameters for on-the-fly pre-processing:
	preprocessing_required = True if argv[10] == "True" else False		
	flat_end = True if argv[11] == "True" else False		
	half_half = True if argv[12] == "True" else False
		
	half_half_line = int(argv[13])
		
	ext_fov = True if argv[14] == "True" else False
		
	norm_sx = int(argv[17])
	norm_dx = int(argv[18])	
		
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
		
	skip_ringrem = True if argv[19] == "True" else False
	ringrem = argv[20]
	
	# Extra reconstruction parameters:
	zerone_mode = True if argv[21] == "True" else False		
	corr_offset = float(argv[22])
		
	reconmethod = argv[23]	
	
	decim_factor = int(argv[24])
	downsc_factor = int(argv[25])
	
	# Parameters for postprocessing:
	postprocess_required = True if argv[26] == "True" else False
	convert_opt = argv[27]
	crop_opt = argv[28]

	# Parameters for on-the-fly phase retrieval:
	phaseretrieval_required = True if argv[29] == "True" else False		
	phrtmethod = int(argv[30])
	phrt_param1 = double(argv[31])   # param1( e.g. regParam, or beta)
	phrt_param2 = double(argv[32])   # param2( e.g. thresh or delta)
	energy = double(argv[33])
	distance = double(argv[34])    
	pixsize = double(argv[35]) / 1000.0 # pixsixe from micron to mm:	
	phrtpad = True if argv[36] == "True" else False
	approx_win = int(argv[37])	

	angles_projfrom = int(argv[38])	
	angles_projto = int(argv[39])	

	preprocessingplan_fromcache = True if argv[40] == "True" else False
	dynamic_ff 	= True if argv[41] == "True" else False

	nr_threads = int(argv[42])	
	tmppath = argv[43]	
	if not tmppath.endswith(sep): tmppath += sep
		
	logfilename = argv[44]		
			
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
	if (sino_idx >= num_sinos):
		sino_idx = num_sinos - 1
	
	# Get correction plan and phase retrieval plan (if required):
	skipflat = False
	
	corrplan = 0	
	im_dark = 0
	EFF = 0
	filtEFF = 0
	if (preprocessing_required):
		if not dynamic_ff:
			# Load flat fielding plan either from cache (if required) or from TDF file and cache it for faster re-use:
			if (preprocessingplan_fromcache):
				try:
					corrplan = cache2plan(infile, tmppath)
				except Exception as e:
					#print "Error(s) when reading from cache"
					corrplan = extract_flatdark(f_in, flat_end, logfilename)
					if (isscalar(corrplan['im_flat']) and isscalar(corrplan['im_flat_after']) ):
						skipflat = True
					else:
						plan2cache(corrplan, infile, tmppath)		
			else:			
				corrplan = extract_flatdark(f_in, flat_end, logfilename)		
				if (isscalar(corrplan['im_flat']) and isscalar(corrplan['im_flat_after']) ):
					skipflat = True
				else:
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

				# Downscale images if necessary:
				im_dark = im_dark[::downsc_factor,::downsc_factor]
				EFF = EFF[::downsc_factor,::downsc_factor,:]	
				filtEFF = filtEFF[::downsc_factor,::downsc_factor,:]	
			
	f_in.close()			

	# Run computation:	
	process( sino_idx, num_sinos, infile, outfile, preprocessing_required, corrplan, skipflat, norm_sx, 
				norm_dx, flat_end, half_half, half_half_line, ext_fov, ext_fov_rot_right, ext_fov_overlap, ringrem, 
				phaseretrieval_required, phrtmethod, phrt_param1, phrt_param2, energy, distance, pixsize, phrtpad, approx_win, angles, 
				angles_projfrom, angles_projto, offset, 
				logtrsf, recpar, circle, scale, overpad, reconmethod, zerone_mode, dset_min, dset_max, decim_factor, 
				downsc_factor, corr_offset, postprocess_required, convert_opt, crop_opt, dynamic_ff, EFF, filtEFF, im_dark, nr_threads, logfilename )		

	# Sample:
	# 311 C:\Temp\BrunGeorgos.tdf C:\Temp\BrunGeorgos.raw 3.1416 -31.0 shepp-logan 1.0 False False True True True True 5 False False 100 0 0 False rivers:11;0 False 0.0 FBP_CUDA 1 1 False - - True 5 1.0 1000.0 22 150 2.2 True 16 0 1799 True True 2 C:\Temp\StupidFolder C:\Temp\log_00.txt



if __name__ == "__main__":
	main(argv[1:])


