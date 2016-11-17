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

# python:
from sys import argv, exit
from os import remove, sep, makedirs, linesep
from os.path import basename, exists
from numpy import finfo, copy, float32, double, amin, amax, tile, concatenate, log as nplog
from numpy import arange, meshgrid, isscalar, ndarray, pi, roll
from time import time
from multiprocessing import Process, Lock

# pystp-specific:
from preprocess.extfov_correction import extfov_correction
from preprocess.flat_fielding import flat_fielding
from preprocess.ring_correction import ring_correction
from preprocess.extract_flatdark import extract_flatdark, _medianize
from preprocess.dynamic_flatfielding import dff_prepare_plan, dynamic_flat_fielding

from reconstruct.rec_astra import recon_astra_fbp, recon_astra_iterative
from reconstruct.rec_fista_tv import recon_fista_tv
from reconstruct.rec_mr_fbp import recon_mr_fbp
from reconstruct.rec_gridrec import recon_gridrec

from postprocess.postprocess import postprocess

from utils.padding import upperPowerOfTwo, padImage, padSmoothWidth

from tifffile import imread, imsave
from h5py import File as getHDF5
import io.tdf as tdf


def reconstruct(im, angles, offset, logtransform, param1, circle, scale, pad, method, rolling, roll_shift,
				zerone_mode, dset_min, dset_max, decim_factor, downsc_factor, corr_offset):
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

	Example (using tiffile.py)
	--------------------------
	>>> # Read input (uncorrected) sinogram
	>>> sino_im1 = imread('sino_0050.tif')
	>>>
	>>> # Get flat and dark correction images:
	>>> im_dark = medianize("\project\tomo", "dark*.tif")
	>>> im_flat = medianize("\project\tomo", "flat*.tif")
	>>>
	>>> # Perform flat fielding and normalization:
	>>> sino_im = normalize(sino_im1, (10,10), (0,0), im_dark, im_flat, 50)    
	>>>  
	>>> # Actual reconstruction: 
	>>> out = reconstruct_fbp(sino_im, -3.0)   
	>>> 
	>>> # Save output slice:
	>>> imsave('slice_0050.tif', out)   	
	
	"""
		
	# Copy images and ensure they are of type float32:
	#im_f = copy(im.astype(float32))   
	im_f = im.astype(float32)

	# Decimate projections if required:
	if decim_factor > 1:
		im_f = im_f[::decim_factor,:]	
	
	# Upscale projections (if required):
	if (abs(scale - 1.0) > finfo(float32).eps):		
		siz_orig1 = im_f.shape[1]		
		im_f = imresize(im_f, (im_f.shape[0], int(round(scale * im_f.shape[1]))), interp='bicubic', mode='F')
		offset = int(offset * scale)		
			
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
	if downsc_factor > 1:
		im_f = im_f[:,::downsc_factor]	
		
	# Sinogram rolling (if required).  It doesn't make sense in limited angle tomography, so check if 180 or 360:
	if ((rolling == True) and (roll_shift > 0)):
		if ( (angles - pi) < finfo(float32).eps ):
			# Flip the last rows:
			im_f[-roll_shift:,:] = im_f[-roll_shift:,::-1]
			# Now roll the sinogram:
			im_f = roll(im_f, roll_shift, axis=0)
		elif ((angles - pi*2.0) < finfo(float32).eps):	
			# Only roll the sinogram:
			im_f = roll(im_f, roll_shift, axis=0)		
			
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
	
	# Perform the actual reconstruction:
	if (method.startswith('FBP')):
		im_f = recon_astra_fbp(im_f, angles, method, param1)	
	elif (method == 'MR-FBP_CUDA'):
		im_f = recon_mr_fbp(im_f, angles)
	elif (method == 'FISTA-TV_CUDA'):
		im_f = recon_fista_tv(im_f, angles, param1, param1)
	else:
		im_f = recon_astra_iterative(im_f, angles, method, param1, zerone_mode)	

		
	# Crop:
	if (pad):		
		im_f = im_f[marg:dim_o + marg, marg:dim_o + marg]			

	# Resize (if necessary):
	if (abs(scale - 1.0) > finfo(float32).eps):
		im_f = imresize(im_f, (siz_orig1, siz_orig1), interp='nearest', mode='F')

	# Return output:
	return im_f.astype(float32)

def reconstruct_gridrec(im1, im2, angles, offset, logtransform, param1, circle, scale, pad, rolling, roll_shift,
				zerone_mode, dset_min, dset_max, decim_factor, downsc_factor, corr_offset):
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

	Example (using tiffile.py)
	--------------------------
	>>> # Read input (uncorrected) sinogram
	>>> sino_im1 = imread('sino_0050.tif')
	>>>
	>>> # Get flat and dark correction images:
	>>> im_dark = medianize("\project\tomo", "dark*.tif")
	>>> im_flat = medianize("\project\tomo", "flat*.tif")
	>>>
	>>> # Perform flat fielding and normalization:
	>>> sino_im = normalize(sino_im1, (10,10), (0,0), im_dark, im_flat, 50)    
	>>>  
	>>> # Actual reconstruction: 
	>>> out = reconstruct_fbp(sino_im, -3.0)   
	>>> 
	>>> # Save output slice:
	>>> imsave('slice_0050.tif', out)   	
	
	"""		
	# Ensure images are of type float32:
	im_f1 = im1.astype(float32)   
	im_f2 = im2.astype(float32)   

	# Decimate projections if required:
	if decim_factor > 1:
		im_f1 = im_f1[::decim_factor,:]	
		im_f2 = im_f2[::decim_factor,:]
	
	# Upscale projections (if required):
	if (abs(scale - 1.0) > finfo(float32).eps):		
		siz_orig1 = im_f.shape[1]		
		im_f1 = imresize(im_f1, (im_f1.shape[0], int(round(scale * im_f1.shape[1]))), interp='bicubic', mode='F')
		im_f2 = imresize(im_f2, (im_f2.shape[0], int(round(scale * im_f2.shape[1]))), interp='bicubic', mode='F')
		offset = int(offset * scale)		
			
	# Apply transformation for changes in the center of rotation:
	if (offset != 0):
		if (offset >= 0):
			im_f1 = im_f1[:,:-offset]
			
			tmp = im_f1[:,0] # Get first column
			tmp = tile(tmp, (offset,1)) # Replicate the first column the right number of times
			im_f1 = concatenate((tmp.T,im_f1), axis=1) # Concatenate tmp before the image

			im_f2 = im_f2[:,:-offset]
			
			tmp = im_f2[:,0] # Get first column
			tmp = tile(tmp, (offset,1)) # Replicate the first column the right number of times
			im_f2 = concatenate((tmp.T,im_f2), axis=1) # Concatenate tmp before the image
						
		else:
			im_f1 = im_f1[:,abs(offset):] 	
			
			tmp = im_f1[:,im_f1.shape[1] - 1] # Get last column
			tmp = tile(tmp, (abs(offset),1)) # Replicate the last column the right number of times
			im_f1 = concatenate((im_f1,tmp.T), axis=1) # Concatenate tmp after the image	

			im_f2 = im_f2[:,abs(offset):] 	
			
			tmp = im_f2[:,im_f2.shape[1] - 1] # Get last column
			tmp = tile(tmp, (abs(offset),1)) # Replicate the last column the right number of times
			im_f2 = concatenate((im_f2,tmp.T), axis=1) # Concatenate tmp after the image	
	
	# Downscale projections (without pixel averaging):
	if downsc_factor > 1:
		im_f1 = im_f1[:,::downsc_factor]			
		im_f2 = im_f2[:,::downsc_factor]	

	# Sinogram rolling (if required).  It doesn't make sense in limited angle tomography, so check if 180 or 360:
	if ((rolling == True) and (roll_shift > 0)):
		if ( (angles - pi) < finfo(float32).eps ):
			# Flip the last rows:
			im_f1[-roll_shift:,:] = im_f1[-roll_shift:,::-1]
			im_f2[-roll_shift:,:] = im_f2[-roll_shift:,::-1]
			# Now roll the sinogram:
			im_f1 = roll(im_f1, roll_shift, axis=0)
			im_f2 = roll(im_f2, roll_shift, axis=0)
		elif ((angles - pi*2.0) < finfo(float32).eps):	
			# Only roll the sinogram:
			im_f1 = roll(im_f1, roll_shift, axis=0)		
			im_f2 = roll(im_f2, roll_shift, axis=0)			
			
	# Scale image to [0,1] range (if required):
	if (zerone_mode):
		
		#print dset_min
		#print dset_max
		#print numpy.amin(im_f[:])
		#print numpy.amax(im_f[:])
		#im_f = (im_f - dset_min) / (dset_max - dset_min)
		
		# Cheating the whole process:
		im_f1 = (im_f1 - numpy.amin(im_f1[:])) / (numpy.amax(im_f1[:]) - numpy.amin(im_f1[:]))
		im_f2 = (im_f2 - numpy.amin(im_f2[:])) / (numpy.amax(im_f2[:]) - numpy.amin(im_f2[:]))		
	
	
	# Apply log transform:
	if (logtransform == True):						
		im_f1[im_f1 <= finfo(float32).eps] = finfo(float32).eps
		im_f1 = -nplog(im_f1 + corr_offset)	

		im_f2[im_f2 <= finfo(float32).eps] = finfo(float32).eps
		im_f2 = -nplog(im_f2 + corr_offset)	
	
	# Replicate pad image to double the width:
	if (pad):	

		dim_o = im_f1.shape[1]		
		n_pad = im_f1.shape[1] + im_f1.shape[1] / 2					
		marg  = (n_pad - dim_o) / 2	

		# Pad image:
		im_f1 = padSmoothWidth(im_f1, n_pad)	
		im_f2 = padSmoothWidth(im_f2, n_pad)		
	
	# Perform the actual reconstruction:	
	[im_f1, im_f2] = recon_gridrec(im_f1, im_f2, angles, param1) 
	
		
	# Crop:
	if (pad):		
		im_f1 = im_f1[marg:dim_o + marg, marg:dim_o + marg]		
		im_f2 = im_f2[marg:dim_o + marg, marg:dim_o + marg]	
		
	# Resize (if necessary):
	if (abs(scale - 1.0) > finfo(float32).eps):
		im_f1 = imresize(im_f1, (siz_orig1, siz_orig1), interp='nearest', mode='F')
		im_f2 = imresize(im_f2, (siz_orig1, siz_orig1), interp='nearest', mode='F')

	# Return output:
	return [im_f1.astype(float32), im_f2.astype(float32)]

def write_log(lock, fname, logfilename, cputime, iotime):    	      
	"""To do...

	"""
	lock.acquire()
	try: 
		# Print out execution time:
		log = open(logfilename,"a")
		log.write(linesep + "\t%s reconstructed (CPU: %0.3f sec - I/O: %0.3f sec)." % (basename(fname), cputime, iotime))
		log.close()	

	finally:
		lock.release()	

def write_log_gridrec(lock, fname1, fname2, logfilename, cputime, iotime):    	      
	"""To do...

	"""
	lock.acquire()
	try: 
		# Print out execution time:
		log = open(logfilename,"a")
		log.write(linesep + "\t%s reconstructed (CPU: %0.3f sec - I/O: %0.3f sec)." % (basename(fname1), cputime/2, iotime/2))
		log.write(linesep + "\t%s reconstructed (CPU: %0.3f sec - I/O: %0.3f sec)." % (basename(fname2), cputime/2, iotime/2))
		log.close()	

	finally:
		lock.release()	

def process_gridrec(lock, int_from, int_to, num_sinos, infile, outpath, preprocessing_required, skipflat, corr_plan, 
			norm_sx, norm_dx, flat_end, half_half, 
			half_half_line, ext_fov, ext_fov_rot_right, ext_fov_overlap, ringrem, angles, angles_projfrom, angles_projto,
			offset, logtransform, param1, circle, scale, pad, rolling, roll_shift, zerone_mode, dset_min, dset_max, decim_factor, 
			downsc_factor, corr_offset,	postprocess_required, convert_opt, crop_opt, dynamic_ff, EFF, filtEFF, im_dark, 
			outprefix, logfilename):
	"""To do...

	"""
	# Process the required subset of images:
	for i in range(int_from, int_to + 1, 2):               
		
		# Read two sinograms:
		t0 = time()
		f_in = getHDF5(infile, 'r')
		if "/tomo" in f_in:
			dset = f_in['tomo']
		else: 
			dset = f_in['exchange/data']
		im1 = tdf.read_sino(dset,i).astype(float32)		
		if ( (i + 1) <= (int_to + 1) ):
			im2 = tdf.read_sino(dset,i + 1).astype(float32)		
		else:
			im2 = im1
		f_in.close()
		t1 = time() 	


		# Apply projection removal (if required):
		im1 = im1[angles_projfrom:angles_projto, :]				
		im2 = im2[angles_projfrom:angles_projto, :]				
			
		# Perform the preprocessing of the sinograms (if required):
		if (preprocessing_required):
			if not skipflat:			
				if dynamic_ff:
					# Dynamic flat fielding with downsampling = 2:
					im1 = dynamic_flat_fielding(im1, i, EFF, filtEFF, 2, im_dark, norm_sx, norm_dx)
				else:
					im1 = flat_fielding (im1, i, corr_plan, flat_end, half_half, half_half_line, norm_sx, norm_dx).astype(float32)		
			im1 = extfov_correction (im1, ext_fov, ext_fov_rot_right, ext_fov_overlap)
			if not skipflat:
				im1 = ring_correction (im1, ringrem, flat_end, corr_plan['skip_flat_after'], half_half, half_half_line, ext_fov)
			else:
				im1 = ring_correction (im1, ringrem, False, False, half_half, half_half_line, ext_fov)

			if not skipflat:
				if dynamic_ff:
					# Dynamic flat fielding with downsampling = 2:
					im2 = dynamic_flat_fielding(im2, i, EFF, filtEFF, 2, im_dark, norm_sx, norm_dx)
				else:
					im2 = flat_fielding (im2, i + 1, corr_plan, flat_end, half_half, half_half_line, norm_sx, norm_dx).astype(float32)		
			im2 = extfov_correction (im2, ext_fov, ext_fov_rot_right, ext_fov_overlap)
			if not skipflat and not dynamic_ff:		
				im2 = ring_correction (im2, ringrem, flat_end, corr_plan['skip_flat_after'], half_half, half_half_line, ext_fov)
			else:
				im2 = ring_correction (im2, ringrem, False, False, half_half, half_half_line, ext_fov)
		

		# Actual reconstruction:
		[im1, im2] = reconstruct_gridrec(im1, im2, angles, offset, logtransform, param1, circle, scale, pad, rolling, roll_shift,
						zerone_mode, dset_min, dset_max, decim_factor, downsc_factor, corr_offset)					

		# Appy post-processing (if required):
		if postprocess_required:
			im1 = postprocess(im1, convert_opt, crop_opt, circle)
			im2 = postprocess(im2, convert_opt, crop_opt, circle)
		else:
			# Create the circle mask for fancy output:
			if (circle == True):
				siz = im1.shape[1]
				if siz % 2:
					rang = arange(-siz / 2 + 1, siz / 2 + 1)
				else:
					rang = arange(-siz / 2,siz / 2)
				x,y = meshgrid(rang,rang)
				z = x ** 2 + y ** 2
				a = (z < (siz / 2 - int(round(abs(offset)/downsc_factor)) ) ** 2)
				
				im1 = im1 * a			
				im2 = im2 * a	
	
		# Write down reconstructed slices:
		t2 = time() 	

		fname1 = outpath + outprefix + '_' + str(i).zfill(4) + '.tif'
		imsave(fname1, im1)

		fname2 = outpath + outprefix + '_' + str(i + 1).zfill(4) + '.tif'
		imsave(fname2, im2)

		t3 = time()
								
		# Write log (atomic procedure - lock used):
		write_log_gridrec(lock, fname1, fname2, logfilename, t2 - t1, (t3 - t2) + (t1 - t0) )		


def process(lock, int_from, int_to, num_sinos, infile, outpath, preprocessing_required, skipflat, corr_plan, norm_sx, norm_dx, 
			flat_end, half_half, 
			half_half_line, ext_fov, ext_fov_rot_right, ext_fov_overlap, ringrem, angles, angles_projfrom, angles_projto,
            offset, logtransform, param1, circle, scale, pad, method, rolling, roll_shift, zerone_mode, dset_min, dset_max, decim_factor, 
			downsc_factor, corr_offset,	postprocess_required, convert_opt, crop_opt, dynamic_ff, EFF, filtEFF, im_dark, 
			outprefix, logfilename):
	"""To do...

	"""
	# Process the required subset of images:
	for i in range(int_from, int_to + 1):                 
		
		# Perform reconstruction (on-the-fly preprocessing and phase retrieval, if required):
		#if (phaseretrieval_required):
			
		#	# Load into memory a bunch of sinograms:
		#	t0 = time()

		#	# Open the TDF file for reading:
		#	f_in = getHDF5(infile, 'r')
		#	if "/tomo" in f_in:
		#		dset = f_in['tomo']
		#	else: 
		#		dset = f_in['exchange/data']

		#	# Prepare the data structure according to the approximation window:
		#	tmp_im = numpy.empty((tdf.get_nr_projs(dset),tdf.get_det_size(dset), approx_win), dtype=float32)

		#	# Load the temporary data structure reading the input TDF file:
		#	# (It can be parallelized Open-MP style)
		#	ct = 0
		#	for j in range(i - approx_win/2, i + approx_win/2 + 1):
		#		if (j < 0):
		#			j = 0
		#		if (j >= num_sinos):
		#			j = num_sinos - 1
		#		a = tdf.read_sino(dset,j).astype(float32)
		#		tmp_im[:,:,ct] = a			
		#		ct = ct + 1
			
		#	# Close the TDF file:	
		#	f_in.close()
		#	t1 = time() 					

		#	# Perform the processing:
		#	if (preprocessing_required):
		#		ct = 0
		#		# (It can be parallelized Open-MP style)
		#		for j in range(i - approx_win/2, i + approx_win/2 + 1):
		#			if (j < 0):
		#				j = 0
		#			if (j >= num_sinos):
		#				j = num_sinos - 1					

		#			tmp_im[:,:,ct] = flat_fielding (tmp_im[:,:,ct], j, corr_plan, flat_end, half_half, half_half_line, norm_sx, norm_dx).astype(float32)			
		#			tmp_im[:,:,ct] = extfov_correction (tmp_im[:,:,ct], ext_fov, ext_fov_rot_right, ext_fov_overlap).astype(float32)			
		#			tmp_im[:,:,ct] = ring_correction (tmp_im[:,:,ct], ringrem, flat_end, corr_plan['skip_flat_after'], half_half, half_half_line, ext_fov).astype(float32)
		#			ct = ct + 1

		#	# Perform phase retrieval:
		#	# (It can be parallelized Open-MP style)
		#	for ct in range(0, tmp_im.shape[0]):

		#		tmp_im[ct,:,:] = phase_retrieval(tmp_im[ct,:,:].T, phrt_plan).astype(float32).T
		#		ct = ct + 1
			
		#	# Extract the central processed sinogram:
		#	im = tmp_im[:,:,approx_win/2]
			
		#else:

		# Read only one sinogram:
		t0 = time()
		f_in = getHDF5(infile, 'r')
		if "/tomo" in f_in:
			dset = f_in['tomo']
		else: 
			dset = f_in['exchange/data']
		im = tdf.read_sino(dset,i).astype(float32)		
		f_in.close()
		t1 = time() 	

		# Apply projection removal (if required):
		im = im[angles_projfrom:angles_projto, :]				
			
		# Perform the preprocessing of the sinogram (if required):
		if (preprocessing_required):
			if not skipflat:
				if dynamic_ff:
					# Dynamic flat fielding with downsampling = 2:
					im = dynamic_flat_fielding(im, i, EFF, filtEFF, 2, im_dark, norm_sx, norm_dx).astype(float32)
				else:
					im = flat_fielding (im, i, corr_plan, flat_end, half_half, half_half_line, norm_sx, norm_dx).astype(float32)		
			im = extfov_correction (im, ext_fov, ext_fov_rot_right, ext_fov_overlap)
			if not skipflat and not dynamic_ff:
				im = ring_correction (im, ringrem, flat_end, corr_plan['skip_flat_after'], half_half, half_half_line, ext_fov)
			else:
				im = ring_correction (im, ringrem, False, False, half_half, half_half_line, ext_fov)
		

		# Actual reconstruction:
		im = reconstruct(im, angles, offset, logtransform, param1, circle, scale, pad, method, rolling, roll_shift,
						zerone_mode, dset_min, dset_max, decim_factor, downsc_factor, corr_offset).astype(float32)			
		
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
				a = (z < (siz / 2 - abs(offset) ) ** 2)
				im = im * a			

		# Write down reconstructed slice:
		t2 = time() 	
		fname = outpath + outprefix + '_' + str(i).zfill(4) + '.tif'
		imsave(fname, im)
		t3 = time()
								
		# Write log (atomic procedure - lock used):
		write_log(lock, fname, logfilename, t2 - t1, (t3 - t2) + (t1 - t0) )


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
	skip_flat = False
	skip_flat_after = True	

	# Get the from and to number of files to process:
	int_from = int(argv[0])
	int_to = int(argv[1])
	   
	# Get paths:
	infile = argv[2]
	outpath = argv[3]

	# Essential reconstruction parameters:
	angles = float(argv[4])
	offset = float(argv[5])
	param1 = argv[6]	
	scale  = int(float(argv[7]))
	
	overpad = True if argv[8] == "True" else False
	logtrsf = True if argv[9] == "True" else False
	circle = True if argv[10] == "True" else False

	outprefix = argv[11]	
	
	# Parameters for on-the-fly pre-processing:
	preprocessing_required = True if argv[12] == "True" else False		
	flat_end = True if argv[13] == "True" else False		
	half_half = True if argv[14] == "True" else False
		
	half_half_line = int(argv[15])
		
	ext_fov = True if argv[16] == "True" else False
		
	norm_sx = int(argv[19])
	norm_dx = int(argv[20])	
		
	ext_fov_rot_right = argv[17]
	if ext_fov_rot_right == "True":
		ext_fov_rot_right = True
		if (ext_fov):
			norm_sx = 0
	else:
		ext_fov_rot_right = False
		if (ext_fov):
			norm_dx = 0
		
	ext_fov_overlap = int(argv[18])
		
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

	angles_projfrom = int(argv[31])	
	angles_projto = int(argv[32])

	rolling = True if argv[33] == "True" else False
	roll_shift = int(argv[34])

	dynamic_ff 	= True if argv[35] == "True" else False
	
	nr_threads = int(argv[36])	
	logfilename = argv[37]	
	process_id = int(logfilename[-6:-4])
	
	# Check prefixes and path:
	#if not infile.endswith(sep): infile += sep
	if not exists(outpath):
		makedirs(outpath)
	
	if not outpath.endswith(sep): outpath += sep
		
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
		log = open(logfilename,"a")
		log.write(linesep + "\tNo projections found. Process will end.")	
		log.close()			
		exit()		

	# Check extrema (int_to == -1 means all files):
	if ((int_to >= num_sinos) or (int_to == -1)):
		int_to = num_sinos - 1
		
	# Log info:
	log = open(logfilename,"w")
	log.write(linesep + "\tInput file: %s" % (infile))	
	log.write(linesep + "\tOutput path: %s" % (outpath))		
	log.write(linesep + "\t--------------")		
	log.write(linesep + "\tPreparing the work plan...")	
	log.close()	
	
	# Get correction plan and phase retrieval plan (if required):
	corrplan = -1
	phrtplan = -1
	
	skipflat = False	

	im_dark = -1
	EFF = -1
	filtEFF = -1
	if (preprocessing_required):
		if not dynamic_ff:
			# Load flat fielding plan either from cache (if required) or from TDF file and cache it for faster re-use:			
			corrplan = extract_flatdark(f_in, flat_end, logfilename)
			if (isscalar(corrplan['im_flat']) and isscalar(corrplan['im_flat_after']) ):
				skipflat = True
			
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
		
	# Log infos:
	log = open(logfilename,"a")
	log.write(linesep + "\tWork plan prepared correctly.")	
	log.write(linesep + "\t--------------")
	log.write(linesep + "\tPerforming reconstruction...")			
	log.close()	

	# Run several threads for independent computation without waiting for threads completion:
	for num in range(nr_threads):
		start = ( (int_to - int_from + 1) / nr_threads)*num + int_from
		if (num == nr_threads - 1):
			end = int_to
		else:
			end = ( (int_to - int_from + 1) / nr_threads)*(num + 1) + int_from - 1
		if (reconmethod == 'GRIDREC'):
			Process(target=process_gridrec, args=(lock, start, end, num_sinos, infile, outpath, preprocessing_required, skipflat, 
						corrplan, norm_sx, norm_dx, flat_end, half_half, half_half_line, ext_fov, ext_fov_rot_right, 
						ext_fov_overlap, ringrem, 
						angles, angles_projfrom, angles_projto, offset, logtrsf, param1, circle, scale, overpad, 
                        rolling, roll_shift,
						zerone_mode, dset_min, dset_max, decim_factor, downsc_factor, corr_offset, 
						postprocess_required, convert_opt, crop_opt, dynamic_ff, EFF, filtEFF, im_dark, outprefix, 
						logfilename )).start()
		else:
			Process(target=process, args=(lock, start, end, num_sinos, infile, outpath, preprocessing_required, skipflat, 
						corrplan, norm_sx, 
						norm_dx, flat_end, half_half, half_half_line, ext_fov, ext_fov_rot_right, ext_fov_overlap, ringrem, 
						angles, angles_projfrom, angles_projto, offset, logtrsf, param1, circle, scale, overpad, 
						reconmethod, rolling, roll_shift,
                        zerone_mode, dset_min, dset_max, decim_factor, downsc_factor, corr_offset, 
						postprocess_required, convert_opt, crop_opt, dynamic_ff, EFF, filtEFF, im_dark, outprefix, 
						logfilename )).start()

	#start = int_from
	#end = int_to
	#if (reconmethod == 'GRIDREC'):
	#	process_gridrec(lock, start, end, num_sinos, infile, outpath, preprocessing_required, skipflat, corrplan, norm_sx, 
	#					norm_dx, flat_end, half_half, half_half_line, ext_fov, ext_fov_rot_right, ext_fov_overlap, ringrem, 
	#					angles, angles_projfrom, angles_projto, offset, logtrsf, param1, circle, scale, overpad, 
	#                   rolling, roll_shift,
	#					zerone_mode, dset_min, dset_max, decim_factor, downsc_factor, corr_offset, 
	#					postprocess_required, convert_opt, crop_opt, dynamic_ff, EFF, filtEFF, im_dark, outprefix, logfilename)
	#else:
	#	process(lock, start, end, num_sinos, infile, outpath, preprocessing_required, skipflat, corrplan, norm_sx, 
	#					norm_dx, flat_end, half_half, half_half_line, ext_fov, ext_fov_rot_right, ext_fov_overlap, ringrem, 
	#					angles, angles_projfrom, angles_projto, offset, logtrsf, param1, circle, scale, overpad, 
	#					reconmethod, rolling, roll_shift, zerone_mode, dset_min, dset_max, decim_factor, downsc_factor, corr_offset, 
	#					postprocess_required, convert_opt, crop_opt, dynamic_ff, EFF, filtEFF, im_dark, outprefix, logfilename)

	# Example:
	# 255 255 C:\Temp\BrunGeorgos.tdf C:\Temp\BrunGeorgos 3.1416 -31.0 shepp-logan 1.0 False False True slice True True True 5 False False 100 0 0 False rivers:11;0 False 0.0 FBP_CUDA 1 1 False - - 0 1799 False 2 C:\Temp\log_00.txt


if __name__ == "__main__":
	main(argv[1:])


