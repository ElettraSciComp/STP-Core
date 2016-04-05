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
import numpy
import time

from time import strftime
from sys import argv, exit
from glob import glob
from h5py import File as getHDF5
import io.tdf as tdf


def _getHISdim ( HISfilename ):
	
	dim1 = 0
	dim2 = 0
	dimz = 0	
	bytecode = numpy.uint16
	
	# Open file:
	try:
		infile = open(HISfilename, "rb")

		# Get file infos:
		tot_bytes = os.path.getsize(HISfilename)

		# Read header:
		Image_tag = infile.read(2)
		Comment_len = numpy.fromstring(infile.read(2), numpy.uint16)[0].astype(numpy.int_)
		dim1 = numpy.fromstring(infile.read(2), numpy.uint16)[0].astype(numpy.int_)
		dim2 = numpy.fromstring(infile.read(2), numpy.uint16)[0].astype(numpy.int_)	
		dim1_offset = numpy.fromstring(infile.read(2), numpy.uint16)[0].astype(numpy.int_)
		dim2_offset = numpy.fromstring(infile.read(2), numpy.uint16)[0].astype(numpy.int_)
		HeaderType = numpy.fromstring(infile.read(2), numpy.uint16)[0]
		Dump = infile.read(50)
		Comment = infile.read(Comment_len)	

		# Set total number of bytes read so far:
		bytes_read = 64 + Comment_len	

		# Set image type:
		bpp = len(numpy.array(0, bytecode).tostring())
		
		# Define chunk size:
		chunksize = dim1 * dim2 * bpp
		
		# Determine number of expected projections:
		dimz = (tot_bytes - bytes_read) / (chunksize + 64) + 1	
		
	finally:
		# Close file:
		infile.close()	
		
	return (dim1, dim2, dimz, bytecode)
	


def _processHIS( HISfilename, dset, dset_offset, provenance_dset, provenance_offset, time_offset, prefix, crop_top, crop_bottom, crop_left, crop_right, logfilename, int_from=0, int_to=-1):

	# Open file:
	infile = open(HISfilename, "rb")

	# Get file infos:
	tot_bytes = os.path.getsize(HISfilename)

	# Read header:
	Image_tag = infile.read(2)
	Comment_len = numpy.fromstring(infile.read(2), numpy.uint16)[0].astype(numpy.int_)
	dim1 = numpy.fromstring(infile.read(2), numpy.uint16)[0].astype(numpy.int_)
	dim2 = numpy.fromstring(infile.read(2), numpy.uint16)[0].astype(numpy.int_)	
	dim1_offset = numpy.fromstring(infile.read(2), numpy.uint16)[0].astype(numpy.int_)
	dim2_offset = numpy.fromstring(infile.read(2), numpy.uint16)[0].astype(numpy.int_)
	HeaderType = numpy.fromstring(infile.read(2), numpy.uint16)[0]
	Dump = infile.read(50)
	Comment = infile.read(Comment_len)	

	# Set total number of bytes read so far:
	bytes_read = 64 + Comment_len	

	# Set image type:
	bytecode = numpy.uint16
	bpp = len(numpy.array(0, bytecode).tostring())
	
	# Define chunk size:
	chunksize = dim1 * dim2 * bpp
	
	# Determine number of expected projections:
	num_proj = (tot_bytes - bytes_read) / (chunksize + 64) + 1	
	
	# Read first projection:
	t1 = time.time()
	block = infile.read(chunksize)
	
	# Convert as numpy array:
	data = numpy.fromstring(block, bytecode)
	im = numpy.reshape( data, [dim2, dim1])	
	im = im[crop_top:im.shape[0]-crop_bottom,crop_left:im.shape[1]-crop_right]	
	
	print numpy.amax(im[:])
	print dset.attrs['max']
	
	# Set minimum and maximum:
	if ( numpy.amin(im[:]) < float(dset.attrs['min']) ):
		dset.attrs['min'] = str(numpy.amin(im[:]))
	if ( numpy.amax(im[:]) > float(dset.attrs['max'])):
		dset.attrs['max'] = str(numpy.amax(im[:]))
		
	print numpy.amax(im[:])
	print dset.attrs['max']
		
	# Check extrema (int_to == -1 means all files) for the projections:
	if ( (int_to >= num_proj) or (int_to <= 0) ):
		int_to = num_proj - 1
	if ( (int_from >= num_proj) or (int_from < 0) ):
		int_from = 0	
		
	# Process first projection (fill HDF5):
	i = 0
	first_index = int(provenance_dset.attrs['first_index'])	
						
	# Save processed image to HDF5 file:												
	#tifffile.imsave('tomo_' + str(i).zfill(4) + '.tif', data)
	if (i >= int_from) and (i <= int_to):	
		tdf.write_tomo(dset, i + dset_offset  - int_from,im)	
		
		# Save provenance metadata:
		t = time.time()	+ time_offset*3600			
		provenance_dset["filename", provenance_offset + i  - int_from] = prefix + '_' + str(i + dset_offset + first_index).zfill(4) 
		provenance_dset["timestamp", provenance_offset + i  - int_from] = numpy.string_(datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
				
		# Print out execution time:	
		t2 = time.time()
		log = open(logfilename,"a")		
		log.write(os.linesep + "\t%s converted in %0.3f sec." % (provenance_dset["filename", provenance_offset + i  - int_from], t2 - t1))			
		log.close()		
	
	# Read all the other projections:	
	try:
		while block:		
				
			# Skip a few bytes:
			t1 = time.time()
			dump = infile.read(64)
			
			# Read the meaningful data:
			block = infile.read(chunksize)						
			
			# Convert as numpy array:
			data = numpy.fromstring(block, bytecode)
			im    = numpy.reshape( data, [dim2, dim1])			
			im    = im[crop_top:im.shape[0]-crop_bottom,crop_left:im.shape[1]-crop_right]	
									
			# Set minimum and maximum:
			if ( float(numpy.amin(im[:])) < float(dset.attrs['min']) ):				
				dset.attrs['min'] = str(numpy.amin(im[:]))				
			if ( float(numpy.amax(im[:])) > float(dset.attrs['max'])):				
				dset.attrs['max'] = str(numpy.amax(im[:]))				
			
			# Process first projection (fill HDF5):
			i = i + 1
								
			# Save processed image to HDF5 file:												
			#tifffile.imsave('tomo_' + str(i).zfill(4) + '.tif', data)
			if (i >= int_from) and (i <= int_to):				
				tdf.write_tomo( dset, i + dset_offset - int_from,im )	
				
				# Save provenance metadata:
				t = time.time()	+ time_offset*3600			
				provenance_dset["filename", provenance_offset + i  - int_from] = prefix + '_' + str(i + dset_offset + first_index  - int_from).zfill(4)
				provenance_dset["timestamp", provenance_offset + i  - int_from] = numpy.string_(datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
							
				# Print out execution time:	
				t2 = time.time()
				log = open(logfilename,"a")		
				log.write(os.linesep + "\t%s converted in %0.3f sec." % (provenance_dset["filename", provenance_offset + i  - int_from], t2 - t1))			
				log.close()	
			
	except  Exception, e:	
		#log = open(logfilename,"a")		
		#log.write(str(e))			
		#log.close()
		pass
	
	finally: 
		# Close file:
		infile.close()

	return provenance_offset + i + 1


def main(argv):          
	"""
	Converts a set of HIS files into a TDF file (HDF5 Tomo Data Format).
	    
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
		
	data_in_path : string
		path of the HIS file of the projections (e.g. "Z:\\sample1.his").
	
	dark_in_path : string
		path of the HIS file of the flat (e.g. "Z:\\sample1_dark.his").
		
	flat_in_path : string
		path of the HIS file of the flat (e.g. "Z:\\sample1_flat.his").
			
	postdark_in_path : string
		path of the HIS file of the flat (e.g. "Z:\\sample1_postdark.his").
		
	postflat_in_path : string
		path of the HIS file of the flat (e.g. "Z:\\sample1_postflat.his").
		
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

	privilege_sino : boolean string
		specify the string "True" if the TDF will privilege a fast read/write of sinograms (the most common 
		case), "False" for fast read/write of projections.
		
	compression : scalar, integer
		an integer value in the range of [1,9] to be used as GZIP compression factor in the HDF5 file, where
		1 is the minimum compression (and maximum speed) and 9 is the maximum (and slow) compression.
		The value 0 can be specified with the meaning of no compression.
		
	log_file : string
		path with filename of a log file (e.g. "R:\\log.txt") where info about the conversion is reported.

	Returns
	-------
	no return value
		
	Example
	-------
	Example call to convert all the tomo*.tif* projections to a TDF with no cropping and minimum compression:
	
		python his2tdf.py 0 -1 "tomo.his" "dark.his" "flat.his" "postdark.his" "postflat.his" "dataset.tdf" 0 0 0 0 
		True True 1 "S:\\conversion.txt"
	
	Requirements
	-------
	- Python 2.7 with the latest NumPy, SciPy, H5Py.
	- tdf.py
	
	Tests
	-------
	Tested with WinPython-64bit-2.7.6.3 (Windows) and Anaconda 2.1.0 (Linux 64-bit).		
	
	"""	
	
	# Get the from and to number of files to process:
	int_from = int(argv[0])
	int_to = int(argv[1]) # -1 means "all files"
	   
	# Get paths:
	tomo_file = argv[2]
	dark_file = argv[3]
	flat_file = argv[4]
	darkpost_file = argv[5]
	flatpost_file = argv[6]
	
	outfile = argv[7]
	
	crop_top      = int(argv[8])  # 0 for all means "no cropping"
	crop_bottom = int(argv[9])
	crop_left      = int(argv[10])
	crop_right    = int(argv[11])
		
	projorder = argv[12]
	if projorder == "True":
		projorder = True
	else:
		projorder = False
		
	privilege_sino = argv[13]
	if privilege_sino == "True":
		privilege_sino = True
	else:
		privilege_sino = False

	# Get compression factor:
	compr_opts = int(argv[14])
	compressionFlag = True;
	if (compr_opts <= 0):
		compressionFlag = False;
	elif (compr_opts > 9):
		compr_opts = 9		
		
	logfilename = argv[15]		

	# Get the files in inpath:
	log = open(logfilename,"w")	
	log.write(os.linesep + "\tInput HIS files:")	
	log.write(os.linesep + "\t\tProjections: %s" % (tomo_file))
	log.write(os.linesep + "\t\tDark: %s" % (dark_file))
	log.write(os.linesep + "\t\tFlat: %s" % (flat_file))
	log.write(os.linesep + "\t\tPost dark: %s" % (darkpost_file))
	log.write(os.linesep + "\t\tPost flat: %s" % (flatpost_file))
	log.write(os.linesep + "\tOutput TDF file: %s" % (outfile))		
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

	log.write(os.linesep + "\t--------------")	
	log.close()
	
	# Remove a previous copy of output:
	if os.path.exists(outfile):
		log = open(logfilename,"a")
		log.write(os.linesep + "\tWarning: an output file with the same name was overwritten.")
		os.remove(outfile)
		log.close()	
			
	# Check input file:
	if not os.path.exists(tomo_file):		
		log = open(logfilename,"a")
		log.write(os.linesep + "\tError: input HIS file for projections does not exist. Process will end.")				
		log.close()			
		exit()	
		
	# First time get the plan:
	log = open(logfilename,"a")
	log.write(os.linesep + "\tPreparing the working plan...")	
	log.close()
			
	# Get info from projection file:
	dim1, dim2, dimz, dtype = _getHISdim ( tomo_file )	
	
	
	if ( ((int_to - int_from + 1) > 0) and ((int_to - int_from + 1) < dimz) ):
		dimz = int_to - int_from + 1
						
	#dsetshape = (num_files,) + im.shape
	if projorder:			
		#dsetshape = tdf.get_dset_shape(privilege_sino, im.shape[1], im.shape[0], num_files)
		dsetshape = tdf.get_dset_shape(dim1 - crop_left - crop_right, dim2 - crop_top - crop_bottom, dimz)
	else:
		#dsetshape = tdf.get_dset_shape(privilege_sino, im.shape[1], num_files, im.shape[0])
		dsetshape = tdf.get_dset_shape(dim1 - crop_left - crop_right, dim2 - crop_top - crop_bottom, dimz)
		
	f = getHDF5( outfile, 'w' )
	print dsetshape
		
	f.attrs['version'] = '1.0'
	f.attrs['implements'] = "exchange:provenance"
	echange_group  = f.create_group( 'exchange' )			
			
	if (compressionFlag):
		dset = f.create_dataset('exchange/data', dsetshape, dtype, chunks=tdf.get_dset_chunks(dim1 - crop_left - crop_right), compression="gzip", compression_opts=compr_opts, shuffle=True, fletcher32=True)
	else:
		dset = f.create_dataset('exchange/data', dsetshape, dtype, chunks=tdf.get_dset_chunks(dim1 - crop_left - crop_right))		

	if privilege_sino:			
		dset.attrs['axes'] = "y:theta:x"
	else:
		dset.attrs['axes'] = "theta:y:x"
			
	dset.attrs['min'] = str(numpy.iinfo(dtype).max)
	dset.attrs['max'] = str(numpy.iinfo(dtype).min)

	# Get the total number of files to consider:
	num_darks = 0
	num_flats = 0
	num_postdarks = 0
	num_postflats = 0
			
	if os.path.exists(dark_file):	
		dim1, dim2, num_darks, dtype = _getHISdim ( dark_file )	
	if os.path.exists(flat_file):	
		dim1, dim2, num_flats, dtype = _getHISdim ( flat_file )	
	if os.path.exists(darkpost_file):	
		dim1, dim2, num_postdarks, dtype = _getHISdim ( darkpost_file )		
	if os.path.exists(flatpost_file):	
		dim1, dim2, num_postflats, dtype = _getHISdim ( flatpost_file )	
			
	tot_files = dimz + num_darks + num_flats + num_postdarks + num_postflats
				
	# Create provenance dataset:
	provenance_dt    = numpy.dtype([("filename", numpy.dtype("S255")), ("timestamp",  numpy.dtype("S255"))])
	metadata_group  = f.create_group( 'provenance' )
	provenance_dset = metadata_group.create_dataset('detector_output', (tot_files,), dtype=provenance_dt)	
			
	provenance_dset.attrs['tomo_prefix'] = 'tomo';
	provenance_dset.attrs['dark_prefix'] = 'dark';
	provenance_dset.attrs['flat_prefix'] = 'flat';
	provenance_dset.attrs['first_index'] = 1;
			
	# Handle the metadata:
	if (os.path.isfile(os.path.dirname(tomo_file) + os.sep + 'logfile.xml')):
		with open (os.path.dirname(tomo_file) + os.sep + 'logfile.xml', "r") as file:
			xml_command = file.read()
		tdf.parse_metadata(f, xml_command)				

	# Print out about plan preparation:
	first_done = True
	log = open(logfilename,"a")
	log.write(os.linesep + "\tWorking plan prepared succesfully.")	
	log.close()				
		
		
	# Get the data from HIS:
	if (num_darks > 0) or (num_postdarks > 0):
		#dsetshape = (num_files,) + im.shape
		if projorder:			
			#dsetshape = tdf.get_dset_shape(privilege_sino, im.shape[1], im.shape[0], num_files)
			dsetshape = tdf.get_dset_shape(dim1 - crop_left - crop_right, dim2 - crop_top - crop_bottom, num_darks + num_postdarks)
		else:
			#dsetshape = tdf.get_dset_shape(privilege_sino, im.shape[1], num_files, im.shape[0])
			dsetshape = tdf.get_dset_shape(dim1 - crop_left - crop_right, dim2 - crop_top - crop_bottom, num_darks + num_postdarks)
		
		if (compressionFlag):
			darkdset = f.create_dataset('exchange/data_dark', dsetshape, dtype, chunks=tdf.get_dset_chunks(dim1 - crop_left - crop_right), compression="gzip", compression_opts=compr_opts, shuffle=True, fletcher32=True)
		else:
			darkdset = f.create_dataset('exchange/data_dark', dsetshape, dtype, chunks=tdf.get_dset_chunks(dim1 - crop_left - crop_right))		

		if privilege_sino:			
			darkdset.attrs['axes'] = "y:theta:x"
		else:
			darkdset.attrs['axes'] = "theta:y:x"
			
		darkdset.attrs['min'] = str(numpy.iinfo(dtype).max)
		darkdset.attrs['max'] = str(numpy.iinfo(dtype).min)
	else:
		log = open(logfilename,"a")
		log.write(os.linesep + "\tWarning: dark images (if any) not considered.")		
		log.close()		
			
	if (num_flats > 0) or (num_postflats > 0):
		
		#dsetshape = (num_files,) + im.shape
		if projorder:			
			#dsetshape = tdf.get_dset_shape(privilege_sino, im.shape[1], im.shape[0], num_files)
			dsetshape = tdf.get_dset_shape(dim1 - crop_left - crop_right, dim2 - crop_top - crop_bottom, num_flats + num_postflats)
		else:
			#dsetshape = tdf.get_dset_shape(privilege_sino, im.shape[1], num_files, im.shape[0])
			dsetshape = tdf.get_dset_shape(dim1 - crop_left - crop_right, dim2 - crop_top - crop_bottom, num_flats + num_postflats)
		
		if (compressionFlag):
			flatdset = f.create_dataset('exchange/data_white', dsetshape, dtype, chunks=tdf.get_dset_chunks(dim1 - crop_left - crop_right), compression="gzip", compression_opts=compr_opts, shuffle=True, fletcher32=True)
		else:
			flatdset = f.create_dataset('exchange/data_white', dsetshape, dtype, chunks=tdf.get_dset_chunks(dim1 - crop_left - crop_right))		

		if privilege_sino:			
			flatdset.attrs['axes'] = "y:theta:x"
		else:
			flatdset.attrs['axes'] = "theta:y:x"
			
		flatdset.attrs['min'] = str(numpy.iinfo(dtype).max)
		flatdset.attrs['max'] = str(numpy.iinfo(dtype).min)
		
	else:
		log = open(logfilename,"a")
		log.write(os.linesep + "\tWarning: flat images (if any) not considered.")		
		log.close()
			
	# Process the HIS:
	provenance_offset = 0
		
	if num_flats > 0:
		provenance_offset = _processHIS( flat_file, flatdset, 0, provenance_dset, provenance_offset, 
			0, 'flat', crop_top, crop_bottom, crop_left, crop_right, logfilename )	
	if num_postflats > 0:
		provenance_offset = _processHIS( flatpost_file, flatdset, num_flats, provenance_dset, provenance_offset, 
			7, 'flat', crop_top, crop_bottom, crop_left, crop_right, logfilename )

	if num_darks > 0:
		provenance_offset = _processHIS( dark_file, darkdset, 0, provenance_dset, provenance_offset,  
			0, 'dark', crop_top, crop_bottom, crop_left, crop_right, logfilename )	
	if num_postdarks > 0:
		provenance_offset = _processHIS( darkpost_file, darkdset, num_darks, provenance_dset, provenance_offset, 
			7, 'dark', crop_top, crop_bottom, crop_left, crop_right, logfilename )

	provenance_offset = _processHIS( tomo_file, dset, 0, provenance_dset, provenance_offset, 
			0, 'tomo', crop_top, crop_bottom, crop_left, crop_right, logfilename, int_from, int_to )	

	
	# Close TDF:
	f.close()	
	
if __name__ == "__main__":
	main(argv[1:])
