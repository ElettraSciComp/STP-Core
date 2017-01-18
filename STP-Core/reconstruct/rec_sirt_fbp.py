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
import sirtfbp

def recon_sirt_fbp(im, angles, iter, temppath ):
	"""Reconstruct a sinogram with the SIRT-FBP algorithm (Pelt, 2015).

	Parameters
	----------
	im : array_like
		Sinogram image data as numpy array.

    iter : int
		Number of iterations to be used for the computation of SIRT filter.

	angles : double
		Value in radians representing the number of angles of the input sinogram.
	
	"""
	# Create ASTRA geometries:
	vol_geom = astra.create_vol_geom(im.shape[1] , im.shape[1])
	proj_geom = astra.create_proj_geom('parallel', 1.0, im.shape[1], linspace(0,angles,im.shape[0],False))
	proj = astra.create_projector('cuda', proj_geom, vol_geom)
	p = astra.OpTomo(proj)

	# Register plugin with ASTRA
	astra.plugin.register(sirtfbp.plugin)

	# Create the ASTRA projector
	im_rec = p.reconstruct('SIRT-FBP',im, iter, extraOptions={'filter_dir':temppath})
	
	return im_rec.astype(float32)