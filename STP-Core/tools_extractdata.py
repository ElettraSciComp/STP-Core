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


import os
import os.path
import numpy
import time

from sys import argv, exit
from h5py import File as getHDF5
from numpy import float32

import io.tdf as tdf

def main(argv):    
	"""Extract a 2D image (projection or sinogram) from the input TDF file (DataExchange HDF5) and
	creates a 32-bit RAW file to disk.

    Parameters
    ----------
    argv[0] : string
		The absolute path of the input TDF.

	argv[1] : int
		The relative position of the image within the dataset.

	argv[2] : string
		One of the following options: 'tomo', 'sino', 'flat', 'dark'.

	argv[3] : string
		The absolute path of the output 32-bit RAW image file. Filename will be modified by adding 
		image width, image height, minimum and maximum value of the input TDF dataset.

	Example
	-------
	tools_extractdata "S:\\dataset.tdf" 128 tomo "R:\\proj"	

    """
	try:
		#
		# Get input parameters:
		#
		infile   = argv[0]
		index    = int(argv[1]) 
		imtype   = argv[2]
		outfile  = argv[3]		
	
		#
		# Body
		#	
	
		# Check if file exists:
		if not os.path.exists(infile):		
			#log = open(logfilename,"a")
			#log.write(os.linesep + "\tError: input TDF file not found. Process will end.")				
			#log.close()			
			exit()	

		# Open the HDF5 file:

		f = getHDF5( infile, 'r' )
		if (imtype == 'sino'):
			if "/tomo" in f:
				dset = f['tomo']	
			else: 
				dset = f['exchange/data']
			im = tdf.read_sino( dset, index )	
		elif (imtype == 'dark'):
			if "/dark" in f:
				dset = f['dark']	
			else: 
				dset = f['exchange/data_dark']	
			im = tdf.read_tomo( dset, index )
		elif (imtype == 'flat'):
			if "/flat" in f:
				dset = f['flat']	
			else: 
				dset = f['exchange/data_white']	
			im = tdf.read_tomo( dset, index )
		else:
			if "/tomo" in f:
				dset = f['tomo']	
			else: 
				dset = f['exchange/data']	
			im = tdf.read_tomo( dset, index )
				

		min = float(numpy.nanmin(im[:]))
		max = float(numpy.nanmax(im[:]))
			
		# Get global attributes (if any):
		try:
			if ('version' in f.attrs):
				if (f.attrs['version'] == '1.0'):	
					min = float(dset_tomo.attrs['min'])
					max = float(dset_tomo.attrs['max'])			
		except: 
			pass
		
		f.close()
		
		# Cast type:
		im = im.astype(float32)
		
		# Modify file name:
		outfile = outfile + '_' + str(im.shape[1]) + 'x' + str(im.shape[0]) + '_' + str(min) + '$' + str(max)	
		
		# Write RAW data to disk:
		im.tofile(outfile)			
	
	except:				
		
		exit()

if __name__ == "__main__":
	main(argv[1:])
