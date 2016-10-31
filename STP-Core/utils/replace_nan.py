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
# Last modified: Setp, 28th 2016
#

from numpy import isnan, nonzero, reshape, interp

def replace_nan (im):
	"""Correct NaN pixels by inteporlation

	Parameters
	----------
	im : array_like
		Image data as numpy array. 
	
	"""
	# Compensate with line-by-line interpolation (better suited for large dead areas):
	im[ im < 0.0 ] = 0.0
	im_f = im.flatten()
	val, x = isnan(im_f), lambda z: z.nonzero()[0]
	im_f[val] = interp(x(val), x(~val), im_f[~val])

	im = reshape(im_f, (im.shape[1], im.shape[0]), order='F').copy().T

	return im 