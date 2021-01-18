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
import tvtomo


def recon_fista_tv(im, angles, lam, fista_iter, iter, cor_shift):
	"""Reconstruct the input sinogram by using the FISTA-TV algorithm

    Parameters
    ----------
    im : array_like
		Image data (sinogram) as numpy array. 

	angles : double
		Value in radians representing the number of angles of the input sinogram.

	lam : double
		Regularization parameter of the FISTA algorithm.

	fista_iter : int
		Number of iterations of the FISTA algorihtm.

	iter : int
		Number of iterations of the TV minimization.	
	
    """	

	# Create ASTRA geometries:
	vol_geom = astra.create_vol_geom(im.shape[1] , im.shape[1])
	proj_geom = astra.create_proj_geom('parallel', 1.0, im.shape[1], linspace(0, angles, im.shape[0], False))

	# Projection geometry with shifted center of rotation (doesn't work apparently):
	#proj_geom = astra.geom_postalignment(proj_geom, cor_shift);

	# Create the ASTRA projector:
	p = tvtomo.ProjectorASTRA2D(proj_geom,vol_geom)	
	
	# Define parameters and FISTA object that performs reconstruction:
	#lam = 1**-14
	f = tvtomo.FISTA(p, lam, fista_iter)

	# Actual reconstruction (takes time):
	im_rec = f.reconstruct(im.astype(float32), iter)
	
	return im_rec.astype(float32)