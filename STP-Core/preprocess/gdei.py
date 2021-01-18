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
# Last modified: April, 5th 2017
#

def gdei(i1, i2, i3, r1, r2, r3, d1, d2, d3, dd1, dd2, dd3):
	"""Apply Generalized Diffraction Enhanced Imaging (GDEI)

	Parameters
	----------
	i1 : array_like
		Image data (sinogram) as numpy array for the first image of the rocking curve

    i2 : array_like
		Image data (sinogram) as numpy array for the second image of the rocking curve

    i3 : array_like
		Image data (sinogram) as numpy array for the third image of the rocking curve
	
	r1, r2, r3 : scalar float
        Coefficients
    
    d1, d2, d3 : scalar float
        Coefficients
    
    dd1, dd2, dd3 : scalar float
        Coefficients
	
	"""
	

	app = (i1 * (d2 * dd3 - dd2 * d3) - i2 * (d1 * dd3 - dd1 * d3) + i3 * (d1 * dd2 - dd1 * d2)) /  \
		 (r1 * (d2 * dd3 - dd2 * d3) - r2 * (d1 * dd3 - dd1 * d3) + r3 * (d1 * dd2 - dd1 * d2))


	ref = - (i1 * (r2 * dd3 - dd2 * r3) - i2 * (r1 * dd3 - dd1 * r3) + i3 * (r1 * dd2 - dd1 * r2)) / \
		 (i1 * (d2 * dd3 - dd2 * d3) - i2 * (d1 * dd3 - dd1 * d3) + i3 * (d1 * dd2 - dd1 * d2))


	sca = 2 * ((i1 * (r2 * d3 - d2 * r3) - i2 * (r1 * d3 - d1 * r3) + i3 * (r1 * d2 - d1 * r2)) /  \
            (i1 * (d2 * dd3 - dd2 * d3) - i2 * (d1 * dd3 - dd1 * d3) + i3 * (d1 * dd2 - dd1 * d2))) - ref ** 2

	return (app, ref, sca)
