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
from glob import glob
from os import linesep
from os.path import sep, basename, exists
from time import time
from multiprocessing import Process, Lock

from postprocess.postprocess import postprocess
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