﻿###########################################################################
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
import numpy
import time

from time import strftime
from sys import argv, exit
from glob import glob

from tifffile import imread, imsave
from h5py import File as getHDF5
import stpio.tdf as tdf
from multiprocessing import Process, Lock

io_warning = False


def _write_data(lock, im, index, offset, abs_offset, imfilename, timestamp, projorder, tot_files, 
				provenance_dt, outfile, dsetname, outshape, outtype, logfilename, itime, n_images):    	      
	"""To do...

	"""
	lock.acquire()
	try:  

		# Open the HDF5 file to be populated with projections (or sinograms):
		t0 = time.time() 			
		f_out = getHDF5(outfile, 'a')					 
		f_out_dset = f_out.require_dataset(dsetname, outshape, outtype, chunks=tdf.get_dset_chunks(outshape[0])) 
		
		# Write the projection file or sinogram file:
		if projorder:
			tdf.write_tomo(f_out_dset, index - abs_offset, im)
		else:
			tdf.write_sino(f_out_dset, index - abs_offset, im)
					
		# Set minimum and maximum (without Infs and NaNs):
		tmp = im[:].astype(numpy.float32)
		tmp = tmp[numpy.nonzero(numpy.isfinite(tmp))]
		if (numpy.amin(tmp[:]) < float(f_out_dset.attrs['min'])):
			f_out_dset.attrs['min'] = str(numpy.amin(tmp[:]))
		if (numpy.amax(tmp[:]) > float(f_out_dset.attrs['max'])):
			f_out_dset.attrs['max'] = str(numpy.amax(tmp[:]))	
		f_out_dset.attrs['avg'] = str(float(f_out_dset.attrs['avg']) + numpy.mean(tmp[:])/(1.0*n_images) )
			
		# Save provenance metadata:
		provenance_dset = f_out.require_dataset('provenance/detector_output', (tot_files,), dtype=provenance_dt)	
		provenance_dset["filename", offset - abs_offset + index] = numpy.string_(os.path.basename(imfilename))
		provenance_dset["timestamp", offset - abs_offset + index] = numpy.string_(datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
		
		# Close the HDF5:
		f_out.close()	
		t1 = time.time()

		# Print out execution time:
		log = open(logfilename,"a")
		log.write(os.linesep + "\t%s processed (I: %0.3f sec - O: %0.3f sec)." % (os.path.basename(imfilename), itime, t1 - t0))
		log.close()	

	except:

		io_warning = True

		# Print out execution time:
		log = open(logfilename,"a")
		log.write(os.linesep + "\tError when writing to TDF %s. File skipped." % (os.path.basename(imfilename)))
		log.close()	

		pass

	finally:
		lock.release()	

def _process(lock, int_from, int_to, offset, abs_offset, files, projorder, outfile, dsetname, outshape, outtype, 
			crop_top, crop_bottom, crop_left, crop_right, tot_files, provenance_dt, logfilename):
	"""To do...

	"""
	# Process the required subset of images:
	for i in range(int_from, int_to + 1):    
		
		# Read input image:
		t0 = time.time()
		try:
			im = imread(files[i])

			# Crop:
			im = im[crop_top:im.shape[0] - crop_bottom,crop_left:im.shape[1] - crop_right]	

			# Get the timestamp:
			t = os.path.getmtime(files[i])
			t1 = time.time() 					
								
			# Save processed image to HDF5 file (atomic procedure - lock used):
			_write_data(lock, im, i, offset, abs_offset, files[i], t, projorder, tot_files, provenance_dt, 
						outfile, dsetname, outshape, outtype, logfilename, t1 - t0, len(files))
		except:

			io_warning = True

			# Print out execution time:
			log = open(logfilename,"a")
			log.write(os.linesep + "\tError when reading %s. File skipped." % (os.path.basename(files[i])))
			log.close()	

			pass



def main(argv):          
	"""
	Converts a sequence of TIFF files into a TDF file (HDF5 Tomo Data Format).
		
	Parameters
	----------
	from : scalar, integer
		among all the projections (or sinogram) files, a subset of files can be specified, 
		ranging from the parameter "from" to the parameter "to" (see next). In most 
		cases, this parameter is 0.
		
	to : scalar, integer
		among all the projections (or sinogram) files, a subset of files can be specified, 
		ranging from the parameter "from" (see previous parameter) to the parameter 
		"to". If the value -1 is specified, all the projection files will be considered.
		
	in_path : string
		path containing the sequence of TIFF files to consider (e.g. "Z:\\sample1\\tomo\\").
		
	out_file : string
		path with filename of the TDF to create (e.g. "Z:\\sample1.tdf"). WARNING: the program 
		does NOT automatically create non-existing folders and subfolders specified in the path. 
		Moreover, if a file with the same name already exists it will be automatically deleted and 
		overwritten.
		
	crop_top : scalar, integer
		during the conversion, images can be cropped if required. This parameter specifies the number 
		of pixels to crop from the top of the image. Leave 0 for no cropping.
		
	crop_bottom : scalar, integer
		during the conversion, images can be cropped if required. This parameter specifies the number 
		of pixels to crop from the bottom of the image. Leave 0 for no cropping.
		
	crop_left : scalar, integer
		during the conversion, images can be cropped if required. This parameter specifies the number 
		of pixels to crop from the left of the image. Leave 0 for no cropping.
		
	crop_right : scalar, integer
		during the conversion, images can be cropped if required. This parameter specifies the number 
		of pixels to crop from the right of the image. Leave 0 for no cropping.
		
	file_prefix : string
		string to be assumed as the filename prefix of the TIFF files to consider for the projection (or 
		sinogram) files. 	E.g. "tomo" will consider files having name "tomo_0001.tif", "tomo_0002.tif". 
		
	flat_prefix : string
		string to be assumed as the filename prefix of the TIFF files to consider for the flat (white field)
		files. E.g. "flat" will consider files having name "flat_1.tif", "flat_2.tif". If dark or flat files have
		to be skipped the string "-" can be specified.
		
	dark_prefix : string
		string to be assumed as the filename prefix of the TIFF files to consider for the dark (dark field)
		files. E.g. "dark" will consider files having name "dark_1.tif", "dark_2.tif". If dark or flat files have
		to be skipped the string "-" can be specified.
		
	projection_order : boolean string
		specify the string "True" if the TIFF files represent projections (the most common case), "False" 
		for sinograms.
		
	privilege_sino : boolean string
		specify the string "True" if the TDF will privilege a fast read/write of sinograms (the most common 
		case), "False" for fast read/write of projections.
		
	compression : scalar, integer
		an integer value in the range of [1,9] to be used as GZIP compression factor in the HDF5 file, where
		1 is the minimum compression (and maximum speed) and 9 is the maximum (and slow) compression.
		The value 0 can be specified with the meaning of no compression.

	nr_threads : int
		number of multiple threads (actually processes) to consider to speed up the whole conversion process.
		
	log_file : string
		path with filename of a log file (e.g. "R:\\log.txt") where info about the conversion is reported.

	Returns
	-------
	no return value
		
	Example
	-------
	Example call to convert all the tomo*.tif* projections to a TDF with no cropping and minimum compression:
	
		python tiff2tdf.py 0 -1 "Z:\\rawdata\\c_1\\tomo\\" "Z:\\work\\c1_compr9.tdf" 0 0 0 0 tomo flat 
		dark True True 1 "S:\\conversion.txt"
	
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

	lock = Lock()

	# Get the from and to number of files to process:
	int_from = int(argv[0])
	int_to = int(argv[1]) # -1 means "all files"
	   
	# Get paths:
	inpath = argv[2]
	outfile = argv[3]
	
	crop_top = int(argv[4])  # 0 for all means "no cropping"
	crop_bottom = int(argv[5])
	crop_left = int(argv[6])
	crop_right = int(argv[7])

	tomoprefix = argv[8]
	flatprefix = argv[9]   # - means "do not consider flat or darks"
	darkprefix = argv[10]  # - means "do not consider flat or darks"

	if (flatprefix == "-") or (darkprefix == "-"):
		skipflat = True
	else:
		skipflat = False
		
	projorder = True if argv[11] == "True" else False		
	privilege_sino = True if argv[12] == "True" else False

	# Get compression factor:
	compr_opts = int(argv[13])
	compressionFlag = True
	if (compr_opts <= 0):
		compressionFlag = False
	elif (compr_opts > 9):
		compr_opts = 9		
	
	nr_threads = int(argv[14])
	logfilename = argv[15]	
	
	# Check prefixes and path:
	if not inpath.endswith(os.path.sep): inpath += os.path.sep

	# Get the files in inpath:
	log = open(logfilename,"w")	
	log.write(os.linesep + "\tInput path: %s" % (inpath))	
	log.write(os.linesep + "\tOutput TDF file: %s" % (outfile))		
	log.write(os.linesep + "\t--------------")			
	log.write(os.linesep + "\tProjection file prefix: %s" % (tomoprefix))
	log.write(os.linesep + "\tDark file prefix: %s" % (darkprefix))
	log.write(os.linesep + "\tFlat file prefix: %s" % (flatprefix))
	log.write(os.linesep + "\t--------------")			
	log.write(os.linesep + "\tCropping:")
	log.write(os.linesep + "\t\tTop: %d pixels" % (crop_top))
	log.write(os.linesep + "\t\tBottom: %d pixels" % (crop_bottom))
	log.write(os.linesep + "\t\tLeft: %d pixels" % (crop_left))
	log.write(os.linesep + "\t\tRight: %d pixels" % (crop_right))
	if (int_to != -1):
		log.write(os.linesep + "\tThe subset [%d,%d] of the input files will be considered." % (int_from, int_to))
	
	if (projorder):
		log.write(os.linesep + "\tProjection order assumed.")
	else:
		log.write(os.linesep + "\tSinogram order assumed.")
		
	if (privilege_sino):
		log.write(os.linesep + "\tFast I/O for sinograms privileged.")
	else:
		log.write(os.linesep + "\tFast I/O for projections privileged.")
	
	if (compressionFlag):
		log.write(os.linesep + "\tTDF compression factor: %d" % (compr_opts))
	else:
		log.write(os.linesep + "\tTDF compression: none.")
	
	if (skipflat):
		log.write(os.linesep + "\tWarning: flat/dark images (if any) will not be considered.")		
	log.write(os.linesep + "\t--------------")	
	log.close()
	
	# Remove a previous copy of output:
	if os.path.exists(outfile):
		log = open(logfilename,"a")
		log.write(os.linesep + "\tWarning: an output file with the same name was overwritten.")
		os.remove(outfile)
		log.close()		
	
	log = open(logfilename,"a")
	log.write(os.linesep + "\tBrowsing input files...")	
	log.close()
			
	# Pythonic way to get file list:
	if os.path.exists(inpath):
		tomo_files = sorted(glob(inpath + tomoprefix + '*.tif*'))
		num_files = len(tomo_files)
	else:		
		log = open(logfilename,"a")
		log.write(os.linesep + "\tError: input path does not exist. Process will end.")				
		log.close()			
		exit()
	
	if (num_files == 0):
		log = open(logfilename,"a")
		log.write(os.linesep + "\tError: no projection files found. Check input path and file prefixes.")				
		log.close()			
		exit()	

	log = open(logfilename,"a")
	log.write(os.linesep + "\tInput files browsed correctly.")	
	log.close()	

	# Check extrema (int_to == -1 means all files):
	if ((int_to >= num_files) or (int_to == -1)):
		int_from = 0
		int_to = num_files - 1

	# In case of subset specified:
	num_files = int_to - int_from + 1

	# Prepare output HDF5 output (should this be atomic?):
	im = imread(tomo_files[0])	

	# Crop:
	im = im[crop_top:im.shape[0] - crop_bottom,crop_left:im.shape[1] - crop_right]		
				
	log = open(logfilename,"a")
	log.write(os.linesep + "\tPreparing the work plan...")	
	log.close()
						
	#dsetshape = (num_files,) + im.shape
	if projorder:			
		#dsetshape = tdf.get_dset_shape(privilege_sino, im.shape[1], im.shape[0],
		#num_files)
		datashape = tdf.get_dset_shape(im.shape[1], im.shape[0], num_files)
	else:
		#dsetshape = tdf.get_dset_shape(privilege_sino, im.shape[1], num_files,
		#im.shape[0])
		datashape = tdf.get_dset_shape(im.shape[1], num_files, im.shape[0])
			
	if not os.path.isfile(outfile):									
		f = getHDF5(outfile, 'w')
			
		f.attrs['version'] = '1.0'
		f.attrs['implements'] = "exchange:provenance"
		echange_group = f.create_group('exchange')			
			
		if (compressionFlag):
			dset = f.create_dataset('exchange/data', datashape, im.dtype, chunks=tdf.get_dset_chunks(im.shape[1]), 
				compression="gzip", compression_opts=compr_opts, shuffle=True, fletcher32=True)
		else:
			dset = f.create_dataset('exchange/data', datashape, im.dtype)		

		if privilege_sino:			
			dset.attrs['axes'] = "y:theta:x"
		else:
			dset.attrs['axes'] = "theta:y:x"
				
		tmp = im[:].astype(numpy.float32)
		tmp = tmp[numpy.nonzero(numpy.isfinite(tmp))]
		dset.attrs['min'] = str(numpy.amin(tmp[:]))
		dset.attrs['max'] = str(numpy.amax(tmp[:]))
		dset.attrs['avg'] = str(0.0)	

		# Get the total number of files to consider:
		tot_files = num_files	
		if not skipflat:		
			num_flats = len(sorted(glob(inpath + flatprefix + '*.tif*')))			
			num_darks = len(sorted(glob(inpath + darkprefix + '*.tif*')))	
			tot_files = tot_files + num_flats + num_darks
				
		# Create provenance dataset:
		provenance_dt = numpy.dtype([("filename", numpy.dtype("S255")), ("timestamp",  numpy.dtype("S255"))])
		metadata_group = f.create_group('provenance')
		provenance_dset = metadata_group.create_dataset('detector_output', (tot_files,), dtype=provenance_dt)	
				
		provenance_dset.attrs['tomo_prefix'] = tomoprefix
		provenance_dset.attrs['dark_prefix'] = darkprefix
		provenance_dset.attrs['flat_prefix'] = flatprefix
		provenance_dset.attrs['first_index'] = int(tomo_files[0][-8:-4])
			
		# Handle the metadata:
		if (os.path.isfile(inpath + 'logfile.xml')):
			with open(inpath + 'logfile.xml', "r") as file:
				xml_command = file.read()
			tdf.parse_metadata(f, xml_command)
					
		f.close()						

	# Print out about plan preparation:
	log = open(logfilename,"a")
	log.write(os.linesep + "\tWork plan prepared succesfully.")	
	log.close()		
		
	# Get the files in inpath:
	if not skipflat:
	
		#
		# Flat part
		#							
		flat_files = sorted(glob(inpath + flatprefix + '*.tif*'))
		num_flats = len(flat_files)		
			
		if (num_flats > 0):
			
			# Create acquisition group:
			im = imread(flat_files[0])								
			im = im[crop_top:im.shape[0] - crop_bottom,crop_left:im.shape[1] - crop_right]
			
			#flatshape = tdf.get_dset_shape(privilege_sino, im.shape[1], im.shape[0],
			#num_flats)
			flatshape = tdf.get_dset_shape(im.shape[1], im.shape[0], num_flats)
			f = getHDF5(outfile, 'a')	
			if (compressionFlag):
				dset = f.create_dataset('exchange/data_white', flatshape, im.dtype, chunks=tdf.get_dset_chunks(im.shape[1]), 
					compression="gzip", compression_opts=compr_opts, shuffle=True, fletcher32=True)
			else:
				dset = f.create_dataset('exchange/data_white', flatshape, im.dtype)		
				
			tmp = im[:].astype(numpy.float32)
			tmp = tmp[numpy.nonzero(numpy.isfinite(tmp))]		
			dset.attrs['min'] = str(numpy.amin(tmp[:]))
			dset.attrs['max'] = str(numpy.amax(tmp[:]))
			dset.attrs['avg'] = str(0.0)
			
			if privilege_sino:			
				dset.attrs['axes'] = "y:theta:x"
			else:
				dset.attrs['axes'] = "theta:y:x"
			f.close()
			
			#process(lock, 0, num_flats - 1, 0, flat_files, True, outfile,
			#'exchange/data_white', dsetshape, im.dtype,
			#	crop_top, crop_bottom, crop_left, crop_right, tot_files, provenance_dt,
			#	logfilename )
				
		else:	
			log = open(logfilename,"a")
			log.write(os.linesep + "\tWarning: flat files not found.")
			log.close()						
				
		#
		# Dark part
		#	
		dark_files = sorted(glob(inpath + darkprefix + '*.tif*'))
		num_darks = len(dark_files)				
			
		if (num_darks > 0):
			im = imread(dark_files[0])			
			im = im[crop_top:im.shape[0] - crop_bottom,crop_left:im.shape[1] - crop_right]
			
			#darkshape = tdf.get_dset_shape(privilege_sino, im.shape[1], im.shape[0],
			#num_flats)
			darkshape = tdf.get_dset_shape(im.shape[1], im.shape[0], num_darks)
			f = getHDF5(outfile, 'a')	
			if (compressionFlag):
				dset = f.create_dataset('exchange/data_dark', darkshape, im.dtype, chunks=tdf.get_dset_chunks(im.shape[1]), 
					compression="gzip", compression_opts=compr_opts, shuffle=True, fletcher32=True)
			else:
				dset = f.create_dataset('exchange/data_dark', darkshape, im.dtype)	
			
			tmp = im[:].astype(numpy.float32)
			tmp = tmp[numpy.nonzero(numpy.isfinite(tmp))]
			dset.attrs['min'] = str(numpy.amin(tmp))
			dset.attrs['max'] = str(numpy.amax(tmp))
			dset.attrs['avg'] = str(0.0)

			if privilege_sino:			
				dset.attrs['axes'] = "y:theta:x"
			else:
				dset.attrs['axes'] = "theta:y:x"
			f.close()		
			
			#process(lock, 0, num_darks - 1, num_flats, dark_files, True, outfile,
			#'exchange/data_dark', dsetshape, im.dtype,
			#	crop_top, crop_bottom, crop_left, crop_right, tot_files, provenance_dt,
			#	logfilename )

		else:
			
			log = open(logfilename,"a")
			log.write(os.linesep + "\tWarning: dark files not found.")
			log.close()		
		
	# Process the required subset of images:
	if not skipflat:
		flatdark_offset = num_flats + num_darks
	else:
		flatdark_offset = 0

	# Spawn the process for the conversion of flat images:
	if (num_flats > 0):
		Process(target=_process, args=(lock, 0, num_flats - 1, 0, 0, flat_files, True, outfile, 'exchange/data_white', 
			flatshape, im.dtype, crop_top, crop_bottom, crop_left, crop_right, tot_files, provenance_dt, logfilename)).start()

	# Spawn the process for the conversion of dark images:
	if (num_darks > 0):
		Process(target=_process, args=(lock, 0, num_darks - 1, num_flats, 0, dark_files, True, outfile, 'exchange/data_dark', 
			darkshape, im.dtype, crop_top, crop_bottom, crop_left, crop_right, tot_files, provenance_dt, logfilename)).start()

	# Start the process for the conversion of the projections (or sinograms) in a multi-threaded way:
	for num in range(nr_threads):
		start = ((int_to - int_from + 1) / nr_threads) * num + int_from
		if (num == nr_threads - 1):
			end = int_to
		else:
			end = ((int_to - int_from + 1) / nr_threads) * (num + 1) + int_from - 1

		Process(target=_process, args=(lock, start, end, flatdark_offset, int_from, tomo_files, projorder, outfile, 'exchange/data', 
				datashape, im.dtype, crop_top, crop_bottom, crop_left, crop_right, tot_files, provenance_dt, logfilename)).start()
		
	#_process(lock, int_from, int_to, flatdark_offset, int_from, tomo_files,
	#            projorder, outfile, 'exchange/data',
	#		    datashape, im.dtype, crop_top, crop_bottom, crop_left, crop_right,
	#		    tot_files, provenance_dt, logfilename )

if __name__ == "__main__":
	main(argv[1:])
