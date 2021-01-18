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

from numpy import concatenate

import imp, inspect, os
import ringremoval


def ring_correction (im, ringrem, flat_end, skip_flat_after, half_half, half_half_line, ext_fov):
	"""Apply ring artifacts compensation by de-striping the input sinogram.

    Parameters
    ----------
    im : array_like
		Image data (sinogram) as numpy array. 
	
	ringrem : string
		String containing ring removal method and parameters
	
	half_half : bool
		True to separately process the sinogram in two parts
	
	half_half_line : int
		Line number considered to identify the two parts to be processed separately. 
		(This parameter is ignored if half_half is False)
	
	skip_flat_after e ext_fov SERVE???
    
    """
	# Get method and args:
	method, args = ringrem.split(":", 1)

	# The "none" filter means no filtering:
	if (method != "none"):

		# Dinamically load module:
		path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
		str = os.path.join(path, "ringremoval",  method + '.py')
		m = imp.load_source(method, str)
			
		if ( (method == "rivers") or (method == "boinhaibel") ):

			if flat_end and not skip_flat_after and half_half and not ext_fov:	
				im_top    = getattr(m, method)( im[0:half_half_line,:], args)
				im_bottom = getattr(m, method)( im[half_half_line:,:], args)
				im = concatenate((im_top,im_bottom), axis=0)					
			else:
				im = getattr(m, method)(im, args)
	
		else:
		
			# Call the module dynamically:
			im = getattr(m, method)(im, args)

	return im 
