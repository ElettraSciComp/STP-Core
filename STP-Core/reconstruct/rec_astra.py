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

from numpy import float32, linspace

import astra

def recon_astra_fbp(im, angles, method, filter_type):
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
	
	if not (method.endswith("CUDA")):
		proj_id = astra.create_projector('strip', proj_geom, vol_geom) # Only for CPU-based algorithms
	
	# Create a data object for the reconstruction
	rec_id = astra.data2d.create('-vol', vol_geom)	

	# We now re-create the sinogram data object:
	sinogram_id = astra.data2d.create('-sino', proj_geom, im)

	# Create configuration:
	cfg = astra.astra_dict('FBP_CUDA')
	cfg['ReconstructionDataId'] = rec_id
	cfg['ProjectionDataId'] = sinogram_id
	cfg['FilterType'] = filter_type	
	
	#overSampling = True
	#if (overSampling == True):
	#	cfg['option']={}
	#	cfg['option']['PixelSuperSampling'] = 2
	
	if not (method.endswith("CUDA")):
		cfg.ProjectorId = proj_id # Only for CPU-based algorithms

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

def recon_astra_iterative(im, angles, method, iterations, zerone_mode):
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