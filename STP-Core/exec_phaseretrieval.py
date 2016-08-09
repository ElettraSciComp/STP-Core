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
from numpy import float32, double, amin, amax
from time import time
from multiprocessing import Process, Lock
from pyfftw.interfaces.cache import enable as pyfftw_cache_enable, disable as pyfftw_cache_disable
from pyfftw.interfaces.cache import set_keepalive_time as pyfftw_set_keepalive_time

from phaseretrieval.tiehom import tiehom, tiehom_plan
from phaseretrieval.phrt   import phrt, phrt_plan

from h5py import File as getHDF5
import io.tdf as tdf


def _write_data(lock, im, index, outfile, outshape, outtype, logfilename, cputime, itime):    	      

	lock.acquire()
	try:        
		t0 = time() 			
		f_out = getHDF5( outfile, 'a' )					 
		f_out_dset = f_out.require_dataset('exchange/data', outshape, outtype, chunks=tdf.get_dset_chunks(outshape[0])) 
		tdf.write_tomo(f_out_dset,index,im.astype(float32))
					
		# Set minimum and maximum:
		if ( amin(im[:]) < float(f_out_dset.attrs['min']) ):
			f_out_dset.attrs['min'] = str(amin(im[:]))
		if ( amax(im[:]) > float(f_out_dset.attrs['max'])):
			f_out_dset.attrs['max'] = str(amax(im[:]))		
		f_out.close()			
		t1 = time() 

		# Print out execution time:
		log = open(logfilename,"a")
		log.write(linesep + "\ttomo_%s processed (CPU: %0.3f sec - I/O: %0.3f sec)." % (str(index).zfill(4), cputime, t1 - t0 + itime))
		log.close()	

	finally:
		lock.release()


def _process(lock, int_from, int_to, infile, outfile, outshape, outtype, method, plan, logfilename):

	# Process the required subset of images:
	for i in range(int_from, int_to + 1):                 
				
		# Read input image:
		t0 = time()
		f_in = getHDF5(infile, 'r')
		if "/tomo" in f_in:
			dset = f_in['tomo']
		else: 
			dset = f_in['exchange/data']
		im = tdf.read_tomo(dset,i).astype(float32)		
		f_in.close()
		t1 = time() 		

		# Perform phase retrieval (first time also PyFFTW prepares a plan):		
		if (method == 0):
			im = tiehom(im, plan).astype(float32)			
		else:
			im = phrt(im, plan, method).astype(float32)			
		t2 = time() 		
								
		# Save processed image to HDF5 file (atomic procedure - lock used):
		_write_data(lock, im, i, outfile, outshape, outtype, logfilename, t2 - t1, t1 - t0)


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
	int_from = int(argv[0])
	int_to = int(argv[1])
	   
	# Get full paths of input TDF and output TDF:
	infile = argv[2]
	outfile = argv[3]
	
	# Get the phase retrieval parameters:
	method = int(argv[4])
	param1 = double(argv[5])   # e.g. regParam, or beta
	param2 = double(argv[6])   # e.g. thresh or delta
	energy = double(argv[7])
	distance = double(argv[8])    
	pixsize = double(argv[9]) / 1000.0 # pixsixe from micron to mm:	
	pad = True if argv[10] == "True" else False
	
	# Number of threads (actually processes) to use and logfile:
	nr_threads = int(argv[11])
	logfilename = argv[12]		

	# Log infos:
	log = open(logfilename,"w")
	log.write(linesep + "\tInput TDF file: %s" % (infile))	
	log.write(linesep + "\tOutput TDF file: %s" % (outfile))		
	log.write(linesep + "\t--------------")
	if (method == 0):
		log.write(linesep + "\tMethod: TIE-Hom (Paganin et al., 2002)")		
		log.write(linesep + "\t--------------")	
		log.write(linesep + "\tDelta/Beta: %0.1f" % ((param2/param1))	)
	#else:
	#	log.write(linesep + "\tMethod: Projected CTF (Moosmann et al., 2011)")		
	#	log.write(linesep + "\t--------------")	
	#	log.write(linesep + "\tDelta/Beta: %0.1f" % ((param2/param1))	)
	log.write(linesep + "\tEnergy: %0.1f keV" % (energy))
	log.write(linesep + "\tDistance: %0.1f mm" % (distance))
	log.write(linesep + "\tPixel size: %0.3f micron" % (pixsize*1000))
	log.write(linesep + "\t--------------")	
	log.write(linesep + "\tBrowsing input files...")	
	log.close()
	
	# Remove a previous copy of output:
	if exists(outfile):
		remove(outfile)
	
	# Open the HDF5 file:
	f_in = getHDF5(infile, 'r')
	if "/tomo" in f_in:
		dset = f_in['tomo']
	else: 
		dset = f_in['exchange/data']
	num_proj = tdf.get_nr_projs(dset)
	num_sinos = tdf.get_nr_sinos(dset)
	
	if (num_proj == 0):
		log = open(logfilename,"a")
		log.write(linesep + "\tNo projections found. Process will end.")	
		log.close()			
		exit()	
	
	log = open(logfilename,"a")
	log.write(linesep + "\tInput files browsed correctly.")	
	log.close()					

	# Check extrema (int_to == -1 means all files):
	if ( (int_to >= num_proj) or (int_to == -1) ):
		int_to = num_proj - 1

	if ( (int_from < 0) ):
		int_from = 0

	# Prepare the plan:
	log = open(logfilename,"a")
	log.write(linesep + "\tPreparing the working plan...")	
	log.close()			

	im = tdf.read_tomo(dset,0).astype(float32)	
	

	outshape = tdf.get_dset_shape(im.shape[1], im.shape[0], num_proj)			
	f_out = getHDF5(outfile, 'w')
	f_out_dset = f_out.create_dataset('exchange/data', outshape, im.dtype) 
	f_out_dset.attrs['min'] = str(amin(im[:]))
	f_out_dset.attrs['max'] = str(amax(im[:]))
	
	f_out_dset.attrs['version'] = '1.0'
	f_out_dset.attrs['axes'] = "y:theta:x"

	f_in.close()
	f_out.close()
				
	if (method == 0):
		# Paganin's:
		plan = tiehom_plan (im, param1, param2, energy, distance, pixsize, pad)
	else:
		plan = phrt_plan (im, energy, distance, pixsize, param2, param1, method, pad)

	# Run several threads for independent computation without waiting for threads completion:
	for num in range(nr_threads):
		start = (num_proj / nr_threads)*num
		if (num == nr_threads - 1):
			end = num_proj - 1
		else:
			end = (num_proj / nr_threads)*(num + 1) - 1
		Process(target=_process, args=(lock, start, end, infile, outfile, outshape, im.dtype, method, plan, logfilename)).start()

	#start = 0
	#end = num_proj - 1
	#_process(lock, start, end, infile, outfile, outshape, im.dtype, method, plan, logfilename)

	#255 256 C:\Temp\BrunGeorgos_corr.tdf C:\Temp\BrunGeorgos_corr_phrt.tdf 0 1.0 2000.0 22.0 300.0 2.2 False 1 C:\Temp\log_00.txt
	#255 256 C:\Temp\BrunGeorgos_corr.tdf C:\Temp\BrunGeorgos_corr_phrt.tdf 4 2.5 1.0 22.0 300.0 2.2 False 1 C:\Temp\log_00.txt
	
if __name__ == "__main__":
	main(argv[1:])

