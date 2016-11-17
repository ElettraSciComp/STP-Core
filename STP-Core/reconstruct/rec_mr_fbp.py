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
import mrfbp
import mrfbp.ASTRAProjector

def recon_mr_fbp(im, angles):
	"""Reconstruct a sinogram with the Minimum Residual FBP algorithm (Pelt, 2013).

	Parameters
	----------
	im : array_like
		Sinogram image data as numpy array.

	angles : double
		Value in radians representing the number of angles of the input sinogram.
	
	"""
	# Create ASTRA geometries:
	vol_geom = astra.create_vol_geom(im.shape[1] , im.shape[1])
	proj_geom = astra.create_proj_geom('parallel', 1.0, im.shape[1], linspace(0,angles,im.shape[0],False))

	# Create the ASTRA projector:
	p = mrfbp.ASTRAProjector.ASTRAProjector2D(proj_geom,vol_geom)
	
	# Create the MR-FBP Reconstructor:
	rec = mrfbp.Reconstructor(p)

	# Reconstruct the image using MR-FBP:
	im_rec = rec.reconstruct(im)
	
	return im_rec.astype(float32)