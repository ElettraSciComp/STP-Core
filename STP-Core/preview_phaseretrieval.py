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

from sys import argv, exit
from os import remove, sep, linesep
from os.path import exists
from numpy import float32, double, nanmin, nanmax, finfo, ndarray
from time import time
from multiprocessing import Process, Lock
from pyfftw.interfaces.cache import enable as pyfftw_cache_enable, disable as pyfftw_cache_disable
from pyfftw.interfaces.cache import set_keepalive_time as pyfftw_set_keepalive_time

from phaseretrieval.phase_retrieval import phase_retrieval, prepare_plan

from h5py import File as getHDF5
from utils.caching import cache2plan, plan2cache
from preprocess.extract_flatdark import extract_flatdark
import io.tdf as tdf


def main(argv):
	"""To do...

	"""
	lock = Lock()

	skip_flat = True
	first_done = False	
	pyfftw_cache_disable()
	pyfftw_cache_enable()
	pyfftw_set_keepalive_time(1800)	

	# Get the from and to number of files to process:
	idx = int(argv[0])
	   
	# Get full paths of input TDF and output TDF:
	infile = argv[1]
	outfile = argv[2]
	
	# Get the phase retrieval parameters:
	beta = double(argv[3])   # param1( e.g. regParam, or beta)
	delta = double(argv[4])   # param2( e.g. thresh or delta)
	energy = double(argv[5])
	distance = double(argv[6])    
	pixsize = double(argv[7]) / 1000.0 # pixsixe from micron to mm:	
	pad = True if argv[8] == "True" else False
	
	# Tmp path and log file:
	tmppath = argv[9]	
	if not tmppath.endswith(sep): tmppath += sep		
	logfilename = argv[10]		

	
	# Open the HDF5 file:
	f_in = getHDF5(infile, 'r')
	if "/tomo" in f_in:
		dset = f_in['tomo']
	else: 
		dset = f_in['exchange/data']
	num_proj = tdf.get_nr_projs(dset)
	num_sinos = tdf.get_nr_sinos(dset)
	
	# Check if the HDF5 makes sense:
	if (num_proj == 0):
		log = open(logfilename,"a")
		log.write(linesep + "\tNo projections found. Process will end.")	
		log.close()			
		exit()		


	# Get flats and darks from cache or from file:
	try:
		corrplan = cache2plan(infile, tmppath)
	except Exception as e:
		#print "Error(s) when reading from cache"
		corrplan = extract_flatdark(f_in, True, logfilename)
		remove(logfilename)
		plan2cache(corrplan, infile, tmppath)

	# Read projection:
	im = tdf.read_tomo(dset,idx).astype(float32)		
	f_in.close()

	# Apply simple flat fielding (if applicable):
	if (isinstance(corrplan['im_flat_after'], ndarray) and isinstance(corrplan['im_flat'], ndarray) and
		isinstance(corrplan['im_dark'], ndarray) and isinstance(corrplan['im_dark_after'], ndarray)) :	
		if (idx < num_proj/2):
			im = (im - corrplan['im_dark']) / (abs(corrplan['im_flat'] - corrplan['im_dark']) + finfo(float32).eps)
		else:
			im = (im - corrplan['im_dark_after']) / (abs(corrplan['im_flat_after'] - corrplan['im_dark_after']) 
				+ finfo(float32).eps)	
					
	# Prepare plan:
	im = im.astype(float32)
	plan = prepare_plan (im, beta, delta, energy, distance, pixsize, padding=pad)

	# Perform phase retrieval (first time also PyFFTW prepares a plan):		
	im = phase_retrieval(im, plan)
	
	# Write down reconstructed preview file (file name modified with metadata):		
	im = im.astype(float32)
	outfile = outfile + '_' + str(im.shape[1]) + 'x' + str(im.shape[0]) + '_' + str( nanmin(im)) + '$' + str( nanmax(im) )	
	im.tofile(outfile)		
	
if __name__ == "__main__":
	main(argv[1:])

