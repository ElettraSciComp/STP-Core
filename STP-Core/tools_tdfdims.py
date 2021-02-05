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

import datetime
import os
import os.path
import time

from sys import argv, exit
from numpy import float32, float64, float16

from h5py import File as getHDF5
import stpio.tdf as tdf

					
	

def main(argv):          
	"""
	Print dimensions of HDF5 file
	"""
	#
	# Get the parameters:
	#

	infile  = argv[0]

	f = getHDF5( infile, 'r' )
	
	if "/tomo" in f:
		dset = f['tomo']	
	else: 
		dset = f['exchange/data']

	print( "Projections: " + tdf.get_nr_projs(dset))
	print( "Slices: " + tdf.get_nr_sinos(dset))
	print( "DetectorSize: " + tdf.get_det_size(dset))

	
if __name__ == "__main__":
	main(argv[1:])
