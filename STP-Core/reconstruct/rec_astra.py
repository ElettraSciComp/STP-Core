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

from numpy import float32, linspace

import astra

def recon_astra_fbp(im, angles, method, filter_type, cor_shift):
	"""Reconstruct the input sinogram by using the FBP implemented in ASTRA toolbox.

    Parameters
    ----------
    im : array_like
		Image data (sinogram) as numpy array. 

	angles : double
		Value in radians representing the number of angles of the sinogram.

	method : string
		A string with either "FBP" or "FBP_CUDA".

	filter_type : string
		The available options are "ram-lak", "shepp-logan", "cosine", "hamming", "hann", "tukey", 
		"lanczos", "triangular", "gaussian", "barlett-hann", "blackman", "nuttall", "blackman-harris", 
		"blackman-nuttall", "flat-top", "kaiser", "parzen".	

    """
	
	vol_geom = astra.create_vol_geom(im.shape[1], im.shape[1])
	proj_geom = astra.create_proj_geom('parallel', 1.0, im.shape[1], linspace(0, angles, im.shape[0], False))

	# Projection geometry with shifted center of rotation:
	proj_geom = astra.geom_postalignment(proj_geom, cor_shift);
	
	if not (method.endswith("CUDA")):
		proj_id = astra.create_projector('strip', proj_geom, vol_geom) # Only for CPU-based algorithms
	
	# Create a data object for the reconstruction
	rec_id = astra.data2d.create('-vol', vol_geom)	

	# We now re-create the sinogram data object:
	sinogram_id = astra.data2d.create('-sino', proj_geom, im)

	# Create configuration:
	cfg = astra.astra_dict(method)
	cfg['ReconstructionDataId'] = rec_id
	cfg['ProjectionDataId'] = sinogram_id
	cfg['FilterType'] = filter_type	
	
	#overSampling = True
	#if (overSampling == True):
	#	cfg['option']={}
	#	cfg['option']['PixelSuperSampling'] = 2
	
	if not (method.endswith("CUDA")):
		cfg['ProjectorId'] = proj_id # Only for CPU-based algorithms

	# Create and run the algorithm object from the configuration structure
	alg_id = astra.algorithm.create(cfg)
	astra.algorithm.run(alg_id, 1)

	# Get the result
	rec = astra.data2d.get(rec_id)

	# Clean up:
	astra.algorithm.delete(alg_id)
	astra.data2d.delete(rec_id)
	astra.data2d.delete(sinogram_id)
	if not (method.endswith("CUDA")):
		astra.projector.delete(proj_id) # For CPU-based algorithms
	
	return rec

def recon_astra_iterative(im, angles, method, iterations, zerone_mode, cor_shift):
	"""Reconstruct the input sinogram by using one of the iterative algorithms implemented in ASTRA toolbox.

    Parameters
    ----------
    im : array_like
		Image data (sinogram) as numpy array. 

	angles : double
		Value in radians representing the number of angles of the sinogram.

	method : string
		A string with e.g "SIRT" or "SIRT_CUDA" (see ASTRA documentation)

	iterations : int
		Number of iterations for the algebraic technique

	zerone_mode : bool
		True if the input sinogram has been rescaled to the [0,1] range (therefore positivity 
		constraints are applied)
	
	
    """
	if (method == "SART_CUDA") or (method == "SART"):
		iterations = int(round(float(iterations))) * im.shape[0]
		
	vol_geom = astra.create_vol_geom(im.shape[1] , im.shape[1])
	proj_geom = astra.create_proj_geom('parallel', 1.0, im.shape[1], linspace(0,angles,im.shape[0],False))

	# Projection geometry with shifted center of rotation:
	proj_geom = astra.geom_postalignment(proj_geom, cor_shift);
	
	if not (method.endswith("CUDA")):
		proj_id = astra.create_projector('strip', proj_geom, vol_geom) # Only for CPU-based algorithms
	
	# Create a data object for the reconstruction
	rec_id = astra.data2d.create('-vol', vol_geom)

	# We now re-create the sinogram data object:
	sinogram_id = astra.data2d.create('-sino', proj_geom, im)

	# Create configuration:
	cfg = astra.astra_dict(method)
	cfg['ReconstructionDataId'] = rec_id
	cfg['ProjectionDataId'] = sinogram_id
	
	if not (method.endswith("CUDA")):
		cfg['ProjectorId'] = proj_id # Only for CPU-based algorithms
	
	cfg['option'] = {}	
	
	if (method.startswith("SART")):
		cfg['option']['ProjectionOrder'] = 'random'
	#if (zerone_mode) and (method != "CGLS_CUDA"):
		#cfg['option']['MinConstraint'] = 0
		#cfg['option']['MaxConstraint'] = 1	

	#overSampling = False
	#if (overSampling == True):
	#	cfg['option']['DetectorSuperSampling'] = 2
	#	cfg['option']['PixelSuperSampling'] = 2
	

	# Create and run the algorithm object from the configuration structure
	alg_id = astra.algorithm.create(cfg)
	astra.algorithm.run(alg_id, int(round(float(iterations))))

	# Get the result
	rec = astra.data2d.get(rec_id)

	# Clean up:
	astra.algorithm.delete(alg_id)
	astra.data2d.delete(rec_id)
	astra.data2d.delete(sinogram_id)	
	if not (method.endswith("CUDA")):
		astra.projector.delete(proj_id) # For CPU-based algorithms
	
	return rec