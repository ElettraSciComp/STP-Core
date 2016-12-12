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

from sys import argv, exit
from os import remove, sep, linesep
from os.path import exists
from numpy import float32, double, nanmin, nanmax, finfo, ndarray
from time import time
from multiprocessing import Process, Lock
from pyfftw.interfaces.cache import enable as pyfftw_cache_enable, disable as pyfftw_cache_disable
from pyfftw.interfaces.cache import set_keepalive_time as pyfftw_set_keepalive_time

from phaseretrieval.tiehom import tiehom, tiehom_plan
from phaseretrieval.phrt   import phrt, phrt_plan

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
	method = int(argv[3])
	param1 = double(argv[4])   # param1( e.g. regParam, or beta)
	param2 = double(argv[5])   # param2( e.g. thresh or delta)
	energy = double(argv[6])
	distance = double(argv[7])    
	pixsize = double(argv[8]) / 1000.0 # pixsixe from micron to mm:	
	pad = True if argv[9] == "True" else False
	
	# Tmp path and log file:
	tmppath = argv[10]	
	if not tmppath.endswith(sep): tmppath += sep		
	logfilename = argv[11]		

	
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
	if (method == 0):
		# Paganin's:
		plan = tiehom_plan (im, param1, param2, energy, distance, pixsize, pad)		
		im = tiehom(im, plan).astype(float32)	
	else:
		plan = phrt_plan (im, energy, distance, pixsize, param2, param1, method, pad)
		im = phrt(im, plan, method).astype(float32)				
	
	# Write down reconstructed preview file (file name modified with metadata):		
	im = im.astype(float32)
	outfile = outfile + '_' + str(im.shape[1]) + 'x' + str(im.shape[0]) + '_' + str( nanmin(im)) + '$' + str( nanmax(im) )	
	im.tofile(outfile)		
	
if __name__ == "__main__":
	main(argv[1:])