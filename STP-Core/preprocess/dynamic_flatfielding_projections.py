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
# Last modified: August, 16th 2016
#
# Code based on the original MATLAB implementation of Vincent Van Nieuwenhove,
# iMinds-vision lab, University of Antwerp. However, the flat fielding is
# performed in the sinogram domain rather than the originally proposed
# projection domain.
#

from numpy import int_, float32, finfo, gradient, sqrt, ndarray, real, dot, sort
from numpy import std, zeros, cov, diag, mean, sum, ComplexWarning, amin, amax
from numpy import concatenate, tile, median

from numpy.random import randn
from numpy.linalg import eig
from scipy.optimize import fmin

from scipy.ndimage.filters import median_filter
from warnings import simplefilter

from stpio.tdf import get_det_size, get_nr_projs, get_nr_sinos, read_tomo


def _parallelAnalysis(ff, n):

	""" Select the number of components for PCA using parallel analysis.
	
	Parameters
	----------
	ff : array_like
		Flat field data as numpy array. Each flat field is a single row 
		of this matrix, different rows are different observations.

	n : int
		Number of repetitions for parallel analysis.

	Return value
	------------
	V : array_like
		Eigen values.

	numPC : int
		Number of components for PCA.

	"""
	# Disable a warning:
	simplefilter("ignore", ComplexWarning)
	stdEFF = std(ff, axis=1, ddof=1)

	kpTrk = zeros((ff.shape[1], n), dtype=float32)
	stdMat = tile(stdEFF,(ff.shape[1], 1)).T

	for i in range(0, n):
		
		sample = stdMat * (randn(ff.shape[0], ff.shape[1])).astype(float32)		
		D, V = eig(cov(sample, rowvar=False))
		kpTrk[:,i] = sort(D).astype(float32)

	mean_ff_EFF = mean(ff,axis=1)
	
	F = ff - tile(mean_ff_EFF, (ff.shape[1], 1)).T
	D, V = eig(cov(F, rowvar=False))

	# Sort eigenvalues from smallest to largest:
	idx = D.argsort()   
	D = D[idx]
	V = V[:,idx]

	sel = zeros(ff.shape[1], dtype=float32)	
	sel[D > (mean(kpTrk, axis=1) + 2*std(kpTrk, axis=1, ddof=1))] = 1
	numPC = sum(sel).astype(int_)

	return (V, numPC)


def _condTVmean(proj, meanFF, FF, DS):
	
	""" Find the optimal estimates of the coefficients of the eigen flat fields.
	
	"""

	# Downsample images (without interpolation):
	proj = proj[::DS, ::DS]       
	meanFF = meanFF[::DS, ::DS]     
	FF2 = zeros((meanFF.shape[0], meanFF.shape[1], FF.shape[2]), dtype=float32)
	for i in range(0, FF.shape[2]):
		FF2[:,:,i] = FF[::DS, ::DS,i]

	# Optimize coefficients:
	xopt = fmin(func=_f, x0=zeros(FF.shape[2], dtype=float32), args=(proj, meanFF, FF2), disp=False)
	
	return xopt.astype(float32)


def _f(x, proj, meanFF, FF):
	""" Objective function to be minimized.
	
	"""
	FF_eff = zeros((FF.shape[0], FF.shape[1]), dtype=float32)
	
	for i in range(0, FF.shape[2]):		
		FF_eff = FF_eff + x[i]*FF[:,:,i]
	
	corProj = proj / (meanFF + FF_eff + finfo(float32).eps) * mean(meanFF + FF_eff)
	[Gx,Gy] = gradient(corProj) 
	mag  = sqrt(Gx ** 2 + Gy ** 2)
	cost = sum(mag)

	return cost


def dff_prepare_plan(white_dset, repetitions, dark):
	""" Prepare the Eigen Flat Fields (EFFs) and the filtered EFFs to
	be used for dynamic flat fielding.

	(Function to be called once before the actual filtering of each projection).
	
	Parameters
	----------
	white_dset : array_like
		3D matrix where each flat field image is stacked along the 3rd dimension.

	repetitions: int
		Number of iterations to consider during parallel analysis.

	dark : array_like
		Single dark image (perhaps the average of a series) to be subtracted from
		each flat field image. If the images are already dark corrected or dark
		correction is not required (e.g. due to a photon counting detector) a matrix
		of the proper shape with zeros has to be passed.

	Return value
	------------
	EFF : array_like
		Eigen flat fields stacked along the 3rd dimension.

	filtEFF : array_like
		Filtered eigen flat fields stacked along the 3rd dimension.

	Note
	----
	In this implementation all the collected white field images have to be loaded into
	memory and an internal 32-bit copy of the white fields is created. Considering also
	that the method better performs with several (i.e. hundreds) flat field images, this 
	function might raise memory errors.

	References
	----------
	V. Van Nieuwenhove, J. De Beenhouwer, F. De Carlo, L. Mancini, F. Marone, 
	and J. Sijbers, "Dynamic intensity normalization using eigen flat fields 
	in X-ray imaging", Optics Express, 23(11), 27975-27989, 2015.

	"""	
	# Get dimensions of flat-field (or white-field) images:
	num_flats = get_nr_projs(white_dset)/4
	num_rows  = get_nr_sinos(white_dset)
	num_cols  = get_det_size(white_dset)
		
	# Create local copy of white-field dataset:
	tmp_dset = zeros((num_rows * num_cols, num_flats), dtype=float32)
	avg      = zeros((num_rows * num_cols), dtype=float32)
					
	# For all the flat images:
	for i in range(0, tmp_dset.shape[1]):                 
		
		# Read i-th flat image and dark-correct:
		tmp_dset[:,i] =  read_tomo(white_dset,i).astype(float32).flatten()	- dark.astype(float32).flatten()
					
		# Sum the image:
		avg = avg + tmp_dset[:,i]

	# Compute the mean:
	avg = avg / num_flats

	# Subtract mean white-field:
	for i in range(0, tmp_dset.shape[1]): 
		tmp_dset[:,i] = tmp_dset[:,i] - avg
			
	# Calculate the number of Eigen Flat Fields (EFF) to use:
	V, nrEFF = _parallelAnalysis(tmp_dset, repetitions)

	# Compute the EFFs (0-th image is the average "conventional" flat field):
	EFF  = zeros((num_rows, num_cols, nrEFF + 1), dtype=float32)
	EFF[:,:,0] = avg.reshape((num_rows, num_cols))			
	for i in range(0, nrEFF): 		
		EFF[:,:,i + 1] = dot(tmp_dset, V[:,num_flats - (i + 1)]).reshape((num_rows, num_cols))	
		
	# Filter the EFFs:
	filtEFF = zeros((num_rows, num_cols, 1 + nrEFF), dtype=float32)
	for i in range(1, 1 + nrEFF):		
		filtEFF[:,:,i] = median_filter(EFF[:,:,i], 3)		

	return EFF, filtEFF


def dynamic_flat_fielding(im, EFF, filtEFF, downsample, dark):

	""" Apply dynamic flat fielding to input projection image.

	(Function to be called for each projection).
	
	Parameters
	----------
	im : array_like
		The (dark-corrected) projection image to process.
		
	EFF : array_like
		Stack of eigen flat fields as return from the dff_prepare_plan function.

	filtEFF : array_like
		Stack of filtered eigen flat fields as return from dff_prepare_plan.

	downsample: int
		Downsampling factor greater than 2 to be used the estimates of weights.

	dark : array_like
		Single dark image (perhaps the average of a series) to be subtracted from
		each flat field image. If the images are already dark corrected or dark
		correction is not required (e.g. due to a photon counting detector) a matrix
		of the proper shape with zeros has to be passed.

	Return value
	------------
	im : array_like
		Filtered projection.

	References
	----------
	V. Van Nieuwenhove, J. De Beenhouwer, F. De Carlo, L. Mancini, F. Marone, 
	and J. Sijbers, "Dynamic intensity normalization using eigen flat fields 
	in X-ray imaging", Optics Express, 23(11), 27975-27989, 2015.

	"""				
	# Dark correct the input image:
	im = im.astype(float32) - dark.astype(float32)
		
	# Estimate weights for a single projection:
	x = _condTVmean(im, EFF[:,:,0], filtEFF[:,:,1:], downsample)

	# Dynamic flat field correction:
	FFeff = zeros(im.shape, dtype=float32)
	for j in range(0, EFF.shape[2] - 1):
		FFeff = FFeff + x[j] * filtEFF[:,:,j + 1]
	
	# Conventional flat fielding (to get mean value):
	tmp = im / (EFF[:,:,0] + finfo(float32).eps)
	tmp[tmp < finfo(float32).eps] = finfo(float32).eps
	mean_val = mean(tmp)

	# Dynamic flat fielding:
	im = im / (EFF[:,:,0] + FFeff + finfo(float32).eps)

	# Re-normalization of the mean with respect to conventional flat fielding:
	im = im / (mean(im) + finfo(float32).eps) * mean_val
			
	# Quick and dirty compensation for detector afterglow:
	size_ct = 3
	while ((float(amin(im)) < finfo(float32).eps) and (size_ct <= 7)):			
		im_f = median_filter(im, size_ct)
		im[im < finfo(float32).eps] = im_f[im < finfo(float32).eps]								
		size_ct += 2
				
	if (float(amin(im)) < finfo(float32).eps):				
		im[im < finfo(float32).eps] = finfo(float32).eps	

	# Return pre-processed image:
	return im




