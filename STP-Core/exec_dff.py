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
# Last modified: August, 17th 2016
#

from sys import argv, exit
from os import remove, sep,  linesep
from os.path import exists
from numpy import float32, amin, amax, isscalar
from time import time
from multiprocessing import Process, Lock

from preprocess.dynamic_flatfielding import dff_prepare_plan, dynamic_flat_fielding
from preprocess.extract_flatdark import extract_flatdark, _medianize

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

def _process (lock, int_from, int_to, infile, outfile, outshape, outtype, EFF, filtEFF, downs, im_dark, logfilename):

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
	
		# Apply dynamic flat fielding:
		im = dynamic_flat_fielding(im, EFF, filtEFF, downs, im_dark)		
								
		# Save processed image to HDF5 file (atomic procedure - lock used):
		t2 = time() 
		_write_data(lock, im, i, outfile, outshape, outtype, logfilename, t2 - t1, t1 - t0)


def main(argv):       
   
	"""Apply dynamic flat fielding to an input TDF file. A filtered TDF file is 
	produced as output.

    Parameters
    ----------
	argv[0] : int
		(Zero-order) number of first projection to process.

	argv[1] : int
		(Zero-order) number of last projection to process. The value -1 has the 
		meaning of "process all the projections".

    argv[2] : string
		The absolute path of input TDF file to process.

	argv[3] : string
		The absolute path of output TDF file that will be created by the process.
		(If file already exists on disk it will be silenty overwritten).

	argv[4] : int
		Downsampling parameter of the dynamic flat fielding (see reference).

	argv[5] : int
		Repetition parameter of the dynamic flat fielding (see reference).

	argv[6] : int
		Number of parallel threads (actually processes) used to speed up the processing.

    argv[7] : string
		The absolute path of a text file where log information is reported.

	References
	----------
	V. Van Nieuwenhove, J. De Beenhouwer, F. De Carlo, L. Mancini, F. Marone, 
	and J. Sijbers, "Dynamic intensity normalization using eigen flat fields 
	in X-ray imaging", Optics Express, 23(11), 27975-27989, 2015.

	Example
	-------
	The following line applies dynamic flat fielding to all the projections of the 
	existing dataset S:\Temp\in.tdf	and it creates the new file S:\Temp\in_dff.tdf
	(the whole process is split into 3 parallel processes):

	exec_dff.py 0 -1 S:\Temp\in.tdf S:\Temp\in_dff.tdf 2 10 3 R:\Temp\dff_log.txt
	
    """ 
	lock = Lock()
	
	# Get the from and to number of files to process:
	int_from = int(argv[0])
	int_to = int(argv[1])
	   
	# Get paths:
	infile = argv[2]
	outfile = argv[3]
	
	# Algorithm parameters:
	downs = int(argv[4])
	repet = int(argv[5])	
	
	# Nr of threads and log file:
	nr_threads = int(argv[6])
	logfilename = argv[7]	


	# Log input parameters:
	log = open(logfilename,"w")
	log.write(linesep + "\tInput TDF file: %s" % (infile))	
	log.write(linesep + "\tOutput TDF file: %s" % (outfile))		
	log.write(linesep + "\t--------------")	
	log.write(linesep + "\tOpening input dataset...")	
	log.close()
	
	# Remove a previous copy of output:
	if exists(outfile):
		remove(outfile)
	
	# Open the HDF5 file:	
	f_in = getHDF5(infile, 'r')

	# Prepare the work plan for dynamic flat fielding:
	log = open(logfilename,"a")
	log.write(linesep + "\t--------------")
	log.write(linesep + "\tPreparing the work plan...")				
	log.close()

	# Open the HDF5 file:	
	f_in = getHDF5(infile, 'r')
	
	skipflat = False
	skipdark = False
	try:
		if "/tomo" in f_in:
			dset = f_in['tomo']					
			if "/flat" in f_in:
				flat_dset = f_in['flat']
				if "/dark" in f_in:
					im_dark = _medianize(f_in['dark'])
				else:										
					skipdark = True
			else:
				skipflat = True # Nothing to do in this case			
		else: 
			dset = f_in['exchange/data']
			if "/exchange/data_white" in f_in:
				flat_dset = f_in['/exchange/data_white']
				if "/exchange/data_dark" in f_in:
					im_dark = _medianize(f_in['/exchange/data_dark'])
				else:					
					skipdark = True
			else:
				skipflat = True # Nothing to do in this case			
	except:
		skipflat = True
		log = open(logfilename,"a")
		log.write(linesep + "\tError reading input dataset. Process will end.")	
		log.close()			
		exit()

	# Check if the HDF5 makes sense:
	num_proj = tdf.get_nr_projs(dset)
	if (num_proj == 0):
		log = open(logfilename,"a")
		log.write(linesep + "\tNo projections found. Process will end.")	
		log.close()			
		exit()
	
	if skipflat:
		log = open(logfilename,"a")
		log.write(linesep + "\tNo flat field images found. Process will end.")	
		log.close()			
		exit()

	tmp = tdf.read_tomo(dset,0).astype(float32)
	if skipdark:				
		im_dark = zeros(tmp.shape)
		log = open(logfilename,"a")
		log.write(linesep + "\tWarning: No dark field images found.")	
		log.close()

	# Check extrema (int_to == -1 means all files):
	if ((int_to >= num_proj) or (int_to == -1)):
		int_to = num_proj - 1
	if ((int_from >= num_proj) or (int_from < 0)):
		int_from = 0
	
	# Prepare plan:
	EFF, filtEFF = dff_prepare_plan(flat_dset, repet, im_dark)	

	# Get the corrected outshape:	
	outshape = tdf.get_dset_shape(im_dark.shape[1], im_dark.shape[0], num_proj)			
	
	# Create the output HDF5 file:	
	f_out = getHDF5(outfile, 'w')
	f_out_dset = f_out.create_dataset('exchange/data', outshape, tmp.dtype) 
	f_out_dset.attrs['min'] = str(amin(tmp[:]))
	f_out_dset.attrs['max'] = str(amax(tmp[:]))
	f_out_dset.attrs['version'] = '1.0'
	f_out_dset.attrs['axes'] = "y:theta:x"

	f_out.close()
	f_in.close()
		
	# Log infos:
	log = open(logfilename,"a")
	log.write(linesep + "\tWorking plan prepared correctly.")	
	log.write(linesep + "\t--------------")
	log.write(linesep + "\tPerforming dynamic flat fielding...")			
	log.close()	

	# Run several threads for independent computation without waiting for threads completion:
	for num in range(nr_threads):
		start = (num_proj / nr_threads)*num
		if (num == nr_threads - 1):
			end = num_proj - 1
		else:
			end = (num_proj / nr_threads)*(num + 1) - 1
		Process(target=_process, args=(lock, start, end, infile, outfile, outshape, tmp.dtype, EFF, filtEFF, 
				downs, im_dark, logfilename)).start()

	# Single-thread version for debug purposes (comment the previous FOR loop):
	#start = 0
	#end = num_proj - 1
	#_process(lock, start, end, infile, outfile, outshape, tmp.dtype, EFF, filtEFF, downs, im_dark, logfilename)


	
if __name__ == "__main__":
	main(argv[1:])
