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

from tifffile import imread, imsave
from h5py import File as getHDF5
import stpio.tdf as tdf

from multiprocessing import Process, Lock

def _write_log(lock, fname, logfilename, iotime):    	      
	"""To do...

	"""
	lock.acquire()
	try: 
		# Print out execution time:
		log = open(logfilename,"a")
		log.write(os.linesep + "\t%s converted in %0.3f sec." % (os.path.basename(fname), iotime))
		log.close()	

	finally:
		lock.release()	

def _process(lock, int_from, int_to, infile, dset_str, TIFFFormat, projorder, outpath, outprefix, logfilename):
	"""To do...

	"""							
	try:			

		f = getHDF5(infile, 'r')	
		dset = f[dset_str]
				
		# Process the required subset of images:
		for i in range(int_from, int_to + 1):                  			
					
			# Read input image:
			t0 = time.time()

			if projorder:
				im = tdf.read_tomo(dset, i)				
			else:
				im = tdf.read_sino(dset, i)		
				
			# Cast type (if required but it should never occur):
			if (((im.dtype).type is float64) or ((im.dtype).type is float16)):
				im = im.astype(float32, copy=False)
				
			if (TIFFFormat):
				fname = outpath + outprefix + '_' + str(i).zfill(4) + '.tif'
				imsave(fname, im)
			else:
				fname = outpath + outprefix + '_' + str(i).zfill(4) + '_' + str(im.shape[1]) + \
						'x' + str(im.shape[0]) + '_' + str(im.dtype) + '.raw'
				im.tofile(fname)

			t1 = time.time() 

			# Print out execution time:	
			_write_log(lock, fname, logfilename, t1 - t0)
					
		f.close()
				
	except Exception: 
		
		pass					
	

def main(argv):          
	"""
	Converts a TDF file (HDF5 Tomo Data Format) into a sequence of TIFF (uncompressed) files.
	    
	Parameters
	----------
	from : scalar, integer
		among all the projections (or sinogram) data, a subset of the volume can 
		be specified, ranging from the parameter "from" to the parameter "to" 
		(see next). In most cases, this parameter is 0.
		
	to : scalar, integer
		among all the projections (or sinogram) data, a subset of the volume can 
		be specified, ranging from the parameter "from" (see previous parameter) to 
		the parameter "to". If the value -1 is specified, all the projection (or sinogram)
		data will be considered.
		
	in_file : string
		path with filename of the TDF to read from (e.g. "Z:\\sample1.tdf").
		
	out_path : string
		path that will contain the sequence of TIFF files (e.g. "Z:\\sample1\\tomo\\"). WARNING: 
		the program does NOT automatically create non-existing folders and subfolders specified 
		in the path. Moreover, if files with the same name already exist they will be automatically 
		overwritten.
		
	file_prefix : string
		string to be assumed as the filename prefix of the TIFF files to create for the projection (or 
		sinogram) data. E.g. "tomo" will create files having name "tomo_0001.tif", "tomo_0002.tif".
		
	flat_prefix : string
		string to be assumed as the filename prefix of the TIFF files to create for the flat (white field)
		data. E.g. "flat" will create files having name "flat_1.tif", "flat_2.tif". If dark or flat data have
		to be skipped the string "-" can be specified.
		
	dark_prefix : string
		string to be assumed as the filename prefix of the TIFF files to create for the dark (dark field)
		data. E.g. "dark" will create files having name "dark_1.tif", "dark_2.tif". If dark or flat data have
		to be skipped the string "-" can be specified.
		
	projection_order : boolean string
		specify the string "True" to create TIFF files for projections (the most common case), "False" 
		for sinograms.		

	TIFF_format : boolean string
		specify the string "True" to create TIFF files, "False" for RAW files.	

	nr_threads : int
		number of multiple threads (actually processes) to consider to speed up the whole conversion process.
		
	log_file : string
		path with filename of a log file (e.g. "R:\\log.txt") where info about the conversion is reported.

	Returns
	-------
	no return value
		
	Example
	-------
	Example call to convert all the projections data to a sequence of tomo*.tif files:
	
		python tdf2tiff.py 0 -1 "C:\Temp\wet12T4part2.tdf" "C:\Temp\tomo" tomo flat dark True True 3 "C:\Temp\log.txt"
	
	Requirements
	-------
	- Python 2.7 with the latest NumPy, SciPy, H5Py.
	- TIFFFile from C. Gohlke
	- tdf.py
	
	Tests
	-------
	Tested with WinPython-64bit-2.7.6.3 (Windows) and Anaconda 2.1.0 (Linux 64-bit).		

	"""	

	lock = Lock()

	# To be used without flat fielding (just conversion):
	first_done = False	

	# Get the from and to number of files to process:
	int_from = int(argv[0])
	int_to = int(argv[1]) # -1 means "all files"
	   
	# Get paths:
	infile = argv[2]
	outpath = argv[3]
	
	fileprefix = argv[4]
	flatprefix = argv[5]
	darkprefix = argv[6]
	
	if (flatprefix == "-"):
		skipflat = True
	else:
		skipflat = False

	if (darkprefix == "-"):
		skipdark = True
	else:
		skipdark = False

	if (fileprefix == "-"):
		skiptomo = True
	else:
		skiptomo = False
	
	projorder = argv[7]
	if projorder == "True":
		projorder = True
	else:
		projorder = False	
		
	TIFFFormat = argv[8]
	if TIFFFormat == "True":
		TIFFFormat = True
	else:
		TIFFFormat = False	
		
	nr_threads = int(argv[9])
	logfilename = argv[10]
	
	# Check prefixes and path:
	if not outpath.endswith(os.path.sep): outpath += os.path.sep
	
	# Init variables:
	num_files = 0
	num_flats = 0
	num_darks = 0

	# Get the files in infile:
	log = open(logfilename,"w")
	log.write(os.linesep + "\tInput TDF: %s" % (infile))
	if (TIFFFormat):
		log.write(os.linesep + "\tOutput path where TIFF files will be created: %s" % (outpath))		
	else:
		log.write(os.linesep + "\tOutput path where RAW files will be created: %s" % (outpath))		
	log.write(os.linesep + "\t--------------")			
	log.write(os.linesep + "\tFile output prefix: %s" % (fileprefix))
	log.write(os.linesep + "\tFlat images output prefix: %s" % (flatprefix))
	log.write(os.linesep + "\tDark images output prefix: %s" % (darkprefix))
	log.write(os.linesep + "\t--------------")	
	
	if (not (skiptomo)):
		if (int_to != -1):
			log.write(os.linesep + "\tThe subset [%d,%d] of the data will be considered." % (int_from, int_to))
	
		if (projorder):
			log.write(os.linesep + "\tProjection order assumed.")
		else:
			log.write(os.linesep + "\tSinogram order assumed.")
	
		log.write(os.linesep + "\t--------------")	
	log.close()	

	if not os.path.exists(infile):		
		log = open(logfilename,"a")
		log.write(os.linesep + "\tError: input TDF file not found. Process will end.")				
		log.close()			
		exit()	

	# Open the HDF5 file:
	f = getHDF5(infile, 'r')
	
	oldTDF = False
	
	try:		
		dset = f['tomo']			
		oldTDF = True
		
	except Exception:
		
		pass
		
	if not oldTDF:
		
		#try:
			dset = f['exchange/data']
			
		#except Exception:
			
		#	log = open(logfilename,"a")
		#	log.write(os.linesep + "\tError: invalid TDF format.  Process will end.")
		#	log.close()
		#	exit()
	
	if projorder:
		num_files = tdf.get_nr_projs(dset)	
	else:
		num_files = tdf.get_nr_sinos(dset)			
	f.close()
	


	# Get attributes:
	try:
		f = getHDF5(infile, 'r')
		if ('version' in f.attrs) and (f.attrs['version'] == 'TDF 1.0'):	
			log = open(logfilename,"a")
			log.write(os.linesep + "\tTDF version 1.0 found.")
			log.write(os.linesep + "\t--------------")
			log.close()
		f.close()				
			
	except:		
		log = open(logfilename,"a")
		log.write(os.linesep + "\tWarning: TDF version unknown. Some features will not be available.")				
		log.write(os.linesep + "\t--------------")
		log.close()			

	# Check extrema (int_to == -1 means all files):
	if ((int_to >= num_files) or (int_to == -1)):
		int_to = num_files - 1
		


	# Spawn the process for the conversion of flat images:
	if not skipflat:

		f = getHDF5(infile, 'r')
		if oldTDF:
			dset_str = 'flat'
		else:
			dset_str = 'exchange/data_white'
		num_flats = tdf.get_nr_projs(f[dset_str])
		f.close()	

		if (num_flats > 0):
			Process(target=_process, args=(lock, 0, num_flats - 1, infile, dset_str, TIFFFormat, 
											True, outpath, flatprefix, logfilename)).start()
			#_process(lock, 0, num_flats - 1, infile, dset_str, TIFFFormat, projorder,
			#outpath, flatprefix, logfilename)

	# Spawn the process for the conversion of dark images:
	if not skipdark:

		f = getHDF5(infile, 'r')
		if oldTDF:
			dset_str = 'dark'
		else:
			dset_str = 'exchange/data_dark'
		num_darks = tdf.get_nr_projs(f[dset_str])
		f.close()	

		if (num_darks > 0):
			Process(target=_process, args=(lock, 0, num_darks - 1, infile, dset_str, TIFFFormat, 
											True, outpath, darkprefix, logfilename)).start()
			#_process(lock, 0, num_darks - 1, infile, dset_str, TIFFFormat, projorder,
			#outpath, darkprefix, logfilename)

	# Spawn the processes for the conversion of projection or sinogram images:
	if not skiptomo:

		if oldTDF:
			dset_str = 'tomo'
		else:
			dset_str = 'exchange/data'
		
		# Start the process for the conversion of the projections (or sinograms) in a
		# multi-threaded way:
		for num in range(nr_threads):
			start = ((int_to - int_from + 1) / nr_threads) * num + int_from
			if (num == nr_threads - 1):
				end = int_to
			else:
				end = ((int_to - int_from + 1) / nr_threads) * (num + 1) + int_from - 1

			Process(target=_process, args=(lock, start, end, infile, dset_str, TIFFFormat, 
											projorder, outpath, fileprefix, logfilename)).start()
			
			#_process(lock, start, end, infile, dset_str, TIFFFormat, projorder,
			#outpath, fileprefix, logfilename)
	
	if __name__ == "__main__":
	main(argv[1:])
