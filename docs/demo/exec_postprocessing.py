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
from glob import glob
from os import linesep
from os.path import sep, basename, exists
from time import time
from multiprocessing import Process, Lock

# pystp-specific:
from stp_core.postprocess.postprocess import postprocess

from tifffile import imread, imsave

def _write_log(lock, fname, logfilename, cputime, iotime):    	      

	lock.acquire()
	try: 
		# Print out execution time:
		log = open(logfilename,"a")
		log.write(linesep + "\t%s processed (CPU: %0.3f sec - I/O: %0.3f sec)." % (basename(fname), cputime, iotime))
		log.close()	

	finally:
		lock.release()	

def _process(lock, int_from, int_to, files, outpath, convert_opt, crop_opt, outprefix, logfilename):

	# Process the required subset of images:
	for i in range(int_from, int_to + 1):                 
		
		# Read i-th slice:
		t0 = time() 
		im = imread(files[i])
		t1 = time() 
			
		# Post process the image:
		im = postprocess(im, convert_opt, crop_opt)	

		# Write down post-processed slice:
		t2 = time()
		fname = outpath + outprefix + '_' + str(i).zfill(4) + '.tif'
		imsave(fname, im)
		t3 = time()
								
		# Write log (atomic procedure - lock used):
		_write_log(lock, fname, logfilename, t2 - t1, (t3 - t2) + (t1 - t0) )


def main(argv):          
	"""To do...

	Usage
	-----
	

	Parameters
	---------
		   
	Example
	--------------------------
	The following line processes the first ten TIFF files of input path 
	"/home/in" and saves the processed files to "/home/out" with the 
	application of the Boin and Haibel filter with smoothing via a Butterworth
	filter of order 4 and cutoff frequency 0.01:

	reconstruct 0 4 C:\Temp\Dullin_Aug_2012\sino_noflat C:\Temp\Dullin_Aug_2012\sino_noflat\output 
	9.0 10.0 0.0 0.0 0.0 true sino slice C:\Temp\Dullin_Aug_2012\sino_noflat\tomo_conv flat dark

	"""
	lock = Lock()
	# Get the from and to number of files to process:
	int_from = int(argv[0])
	int_to = int(argv[1])
	   
	# Get input and output paths:
	inpath = argv[2]
	outpath = argv[3]
	
	if not inpath.endswith(sep): inpath += sep
	if not outpath.endswith(sep): outpath += sep

	# Get parameters:
	convert_opt = argv[4]
	crop_opt = argv[5]	

	outprefix = argv[6]

	# Number of threads to use and logfile:
	nr_threads = int(argv[7])
	logfilename = argv[8]		

	# Get the files in infile:
	log = open(logfilename,"w")
	log.write(linesep + "\tInput TIFF folder: %s" % (inpath))	
	log.write(linesep + "\tOutput TIFF folder: %s" % (outpath))		
	log.write(linesep + "\t--------------")			
	if (int_to != -1):
		log.write(linesep + "\tThe subset [%d,%d] of the input files will be considered." % (int_from, int_to))
	log.write(linesep + "\tCropping:")
	crop_opt_num = crop_opt.split(":")
	log.write(linesep + "\t\tTop: %s pixels" % (crop_opt_num[0]))
	log.write(linesep + "\t\tBottom: %s pixels" % (crop_opt_num[1]))
	log.write(linesep + "\t\tLeft: %s pixels" % (crop_opt_num[2]))
	log.write(linesep + "\t\tRight: %s pixels" % (crop_opt_num[3]))
	conv_method, conv_args = convert_opt.split(":", 1)
	if (conv_method == "linear8"):
		min, max = conv_args.split(";")   
		log.write(linesep + "\tConversion to 8-bit by remapping range [%s,%s] to [0,255]." % (min, max))
	elif (conv_method == "linear"):
		min, max = conv_args.split(";")   
		log.write(linesep + "\tConversion to 16-bit by remapping range [%s,%s] to [0,65535]." % (min, max))
	log.write(linesep + "\t--------------")	
	log.write(linesep + "\tBrowsing input folder...")	
	log.close()
		
	files = sorted(glob(inpath + '*.tif*'))
	num_files = len(files)		
	
	if ((int_to >= num_files) or (int_to == -1)):
		int_to = num_files - 1
	
	# Log infos:
	log = open(logfilename,"a")
	log.write(linesep + "\tInput folder browsed correctly.")	
	log.close()	

	# Run several threads for independent computation without waiting for threads completion:
	for num in range(nr_threads):
		start = ( (int_to - int_from + 1) / nr_threads)*num + int_from
		if (num == nr_threads - 1):
			end = int_to
		else:
			end = ( (int_to - int_from + 1) / nr_threads)*(num + 1) + int_from - 1
		Process(target=_process, args=(lock, start, end, files, outpath, convert_opt, crop_opt, outprefix, logfilename )).start()

	#start = 0
	#end = num_files - 1
	#process(lock, start, end, files, outpath, convert_opt, crop_opt, outprefix, logfilename )

	#0 -1 C:\Temp\BrunGeorgos C:\Temp\BrunGeorgos\slice_8 linear8:-0.01;0.01 10:10:10:20 slice C:\Temp\log_00_conv.txt


if __name__ == "__main__":
	main(argv[1:])