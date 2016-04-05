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

import datetime
import os
import os.path
import time

from sys import argv, exit
from numpy import float32, float64

from tifffile import imread, imsave
from h5py import File as getHDF5
import io.tdf as tdf

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
		
	log_file : string
		path with filename of a log file (e.g. "R:\\log.txt") where info about the conversion is reported.

	Returns
	-------
	no return value
		
	Example
	-------
	Example call to convert all the projections data to a sequence of tomo*.tif files:
	
		python tdf2tiff.py 0 -1 "C:\Temp\wet12T4part2.tdf" "C:\Temp\tomo" tomo flat dark True True "C:\Temp\log.txt"
	
	Requirements
	-------
	- Python 2.7 with the latest NumPy, SciPy, H5Py.
	- TIFFFile from C. Gohlke's website http://www.lfd.uci.edu/~gohlke/ 
	   (consider also to install TIFFFile.c for performances).
	- tdf.py
	
	Tests
	-------
	Tested with WinPython-64bit-2.7.6.3 (Windows) and Anaconda 2.1.0 (Linux 64-bit).		

	"""	
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
	
	if (flatprefix == "-") or (darkprefix == "-"):
		skipflat = True
	else:
		skipflat = False
	
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
		
	logfilename = argv[9]
	
	# Check prefixes and path:
	if not outpath.endswith(os.path.sep): outpath += os.path.sep
	
	# Init variables:
	num_files = 0
	num_flats = 0
	num_darks = 0

	# Get the files in infile:
	log = open(logfilename,"w")
	log.write(os.linesep + "\tInput TDF: %s" % (infile))
	if ( TIFFFormat ):
		log.write(os.linesep + "\tOutput path where TIFF files will be created: %s" % (outpath))		
	else:
		log.write(os.linesep + "\tOutput path where RAW files will be created: %s" % (outpath))		
	log.write(os.linesep + "\t--------------")			
	log.write(os.linesep + "\tFile output prefix: %s"  % (fileprefix))
	log.write(os.linesep + "\tDark images output prefix: %s" % (flatprefix))
	log.write(os.linesep + "\tFlat images output prefix: %s" % (darkprefix))
	log.write(os.linesep + "\t--------------")	
	if (int_to != -1):
		log.write(os.linesep + "\tThe subset [%d,%d] of the data will be considered." % (int_from, int_to))
	
	if (projorder):
		log.write(os.linesep + "\tProjection order assumed.")
	else:
		log.write(os.linesep + "\tSinogram order assumed.")
	
	if (skipflat):
		log.write(os.linesep + "\tWarning: flat/dark images (if any) will not be considered.")		
	log.write(os.linesep + "\t--------------")	
	log.close()	

	if not os.path.exists(infile):		
		log = open(logfilename,"a")
		log.write(os.linesep + "\tError: input TDF file not found. Process will end.")				
		log.close()			
		exit()	

	# Open the HDF5 file:
	f = getHDF5( infile, 'r' )
	
	oldTDF = False;
	
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
		#	log.write(os.linesep + "\tError: invalid TDF format. Process will end.")					
		#	log.close()			
		#	exit()
	
	if projorder:
		num_files = tdf.get_nr_projs(dset)	
	else:
		num_files = tdf.get_nr_sinos(dset)			
	f.close()
	


	# Get attributes:
	try:
		f = getHDF5( infile, 'r' )
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
	if ( (int_to >= num_files) or (int_to == -1) ):
		int_to = num_files - 1
		
	# Get the files in infile:
	if not skipflat:			
			
		#
		# Flat part
		#								
		try:			

			t0 = time.time() 
			f = getHDF5( infile, 'r' )	

			if oldTDF:	
				dset = f['flat']
			else:
				dset = f['exchange/data_white']			
	
			num_flats = tdf.get_nr_projs(dset)	
				
			#if ('version' in f.attrs) and (f.attrs['version'] == "TDF 1.0"):	
			if ('version' in f.attrs):
				if (f.attrs['version'] == '1.0'):
					provenance_dset = f['provenance/detector_output']
					
			# Process the required subset of images:
			for i in range(0, num_flats):                  			
					
				# Read input image:
				t1 = time.time()
				im = tdf.read_tomo( dset, i )
				if ( TIFFFormat ):
					if ('version' in f.attrs):
						if (f.attrs['version'] == '1.0'):
							if (os.path.splitext(provenance_dset["filename", i])[0] == provenance_dset["filename", i]):
								fname = outpath + provenance_dset["filename", i] + '.tif'
							else:
								fname = outpath + provenance_dset["filename", i]
					else:
						fname = outpath + flatprefix + '_' + str(i + 1).zfill(4) + '.tif'
				else:
					fname = outpath + flatprefix + '_' + str(i + 1).zfill(4) + '_' + str(im.shape[1]) + 'x' + str(im.shape[0]) + '_' + str(im.dtype)	+ '.raw'
				
				# Cast type (if required but it should never occur):		
				if ((im.dtype).type is float64):
					im = im.astype(float32, copy=False)
				
				if ( TIFFFormat ):
					imsave(fname, im)
				else:
					im.tofile(fname)				
				
				try:
					if ('version' in f.attrs):
						if (f.attrs['version'] == '1.0'):															
							t = int(time.mktime(datetime.datetime.strptime(provenance_dset["timestamp", i], "%Y-%m-%d %H:%M:%S.%f").timetuple()))							
							os.utime(fname, (t,t) )
				except:
					pass

				t2 = time.time() 

				# Print out execution time:	
				log = open(logfilename,"a")		
				log.write(os.linesep + "\t%s created in %0.3f sec." % (os.path.basename(fname), t2 - t1))			
				log.close()	
					
			f.close()
				
		except Exception: 
				
			log = open(logfilename,"a")
			log.write(os.linesep + "\tWarning: no dataset named \"flat\" found.")
			log.close()
				
			pass		
		
		#
		# Dark part
		#			
		try:
			
			t0 = time.time() 
			f = getHDF5( infile, 'r' )
			if oldTDF:	
				dset = f['dark']
			else:
				dset = f['exchange/data_dark']			
			num_darks = tdf.get_nr_projs(dset)	
				
			if ('version' in f.attrs):
				if (f.attrs['version'] == '1.0'):	
					provenance_dset = f['provenance/detector_output']
						
			# Process the required subset of images:
			for i in range(0, num_darks):                  
					
				# Read input image:
				t1 = time.time()
				im = tdf.read_tomo( dset, i )
				
				if ( TIFFFormat ):
					if ('version' in f.attrs):
						if (f.attrs['version'] == '1.0'):							
							if (os.path.splitext(provenance_dset["filename", num_flats + i])[0] == provenance_dset["filename", num_flats + i]):
								fname = outpath + provenance_dset["filename", num_flats + i] + '.tif'
							else:
								fname = outpath + provenance_dset["filename", num_flats + i]
					else:
						fname = outpath + darkprefix + '_' + str(i + 1).zfill(4) + '.tif'
				else:
					fname = outpath + darkprefix + '_' + str(i + 1).zfill(4) + '_' + str(im.shape[1]) + 'x' + str(im.shape[0]) + '_' + str(im.dtype)	+ '.raw'
				
				# Cast type (if required but it should never occur):		
				if ((im.dtype).type is float64):
					im = im.astype(float32, copy=False)
					
				if ( TIFFFormat ):
					imsave(fname, im)
				else:
					im.tofile(fname)		
					
				try:
					if ('version' in f.attrs):
						if (f.attrs['version'] == '1.0'):																
							t = int(time.mktime(datetime.datetime.strptime(provenance_dset["timestamp", num_flats + i], "%Y-%m-%d %H:%M:%S.%f").timetuple()))							
							os.utime(fname, (t,t) )
				except:
					pass
						
				t2 = time.time() 

				# Print out execution time:	
				log = open(logfilename,"a")		
				log.write(os.linesep + "\t%s created in %0.3f sec." % (os.path.basename(fname), t2 - t1))			
				log.close()	
					
			f.close()
				
		except Exception: 
				
			log = open(logfilename,"a")
			log.write(os.linesep + "\tWarning: no dataset named \"dark\" found.")
			log.close()
				
			pass		
		
	else:
		num_flats = 0
		num_darks = 0
		
	#
	# Tomo part
	#	
	
	# Read i-th image from input folder:
	t0 = time.time() 
	f = getHDF5( infile, 'r' )
	
	if oldTDF:		
		dset = f['tomo']
	else:
		dset = f['exchange/data']	
	
	if ('version' in f.attrs):
		if (f.attrs['version'] == '1.0'):	
			provenance_dset = f['provenance/detector_output']
	
	if not skipflat:	
		offset = num_flats + num_darks
	else:
		offset = 0
	
	# Process the required subset of images:
	for i in range(int_from, int_to + 1):                  
		
		# Read input image:
		t1 = time.time()
		if projorder:
			im = tdf.read_tomo( dset, i )			
		else:
			im = tdf.read_sino( dset, i )			
		
		
		# Cast type (if required but it should never occur):		
		if ((im.dtype).type is float64):			
			im = im.astype(float32, copy=False)
		
		# Save file:
		if ( TIFFFormat ):
			if ('version' in f.attrs):
				if (f.attrs['version'] == '1.0'):	
					if (os.path.splitext(provenance_dset["filename", offset + i])[0] == provenance_dset["filename", offset + i]):
						fname = outpath + provenance_dset["filename", offset + i] + '.tif'
					else:
						fname = outpath + provenance_dset["filename", offset + i]
			else:
				fname = outpath + fileprefix + '_' + str(i + 1).zfill(4) + '.tif'
			imsave(fname, im)				
		else:
			fname = outpath + fileprefix + '_' + str(i + 1).zfill(4) + '_'	+ str(im.shape[1]) + 'x' + str(im.shape[0]) + '_' + str(im.dtype)	+ '.raw'
			im.tofile(fname)
		
		# Change modified date:
		try:
			if ('version' in f.attrs):
				if (f.attrs['version'] == '1.0'):														
					t = int(time.mktime(datetime.datetime.strptime(provenance_dset["timestamp", offset + i], "%Y-%m-%d %H:%M:%S.%f").timetuple()))							
					os.utime(fname, (t,t) )
		except:
			pass
		
		t2 = time.time() 

		# Print out execution time:	
		log = open(logfilename,"a")		
		log.write(os.linesep + "\t%s created in %0.3f sec." % (os.path.basename(fname), t2 - t1))			
		log.close()	
			
	f.close()		

	
if __name__ == "__main__":
	main(argv[1:])
