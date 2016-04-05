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

from sys import maxint
from os import linesep
from os.path import splitext
from time import time, mktime
from datetime import datetime

from h5py import File as getHDF5
from numpy import float32, ndarray

from io.tdf import get_nr_projs, read_tomo



def _medianize_withprovenance(dset, provenance_dset, tomoprefix, darkorflatprefix, flagafter):	
	
	# Get minimum and max timestamps for projection
	min_t  = maxint
	max_t = -1
	
	for i in range(0, provenance_dset.shape[0]):  
		if provenance_dset["filename", i].startswith(tomoprefix):
			t = int(mktime(datetime.strptime(provenance_dset["timestamp", i], "%Y-%m-%d %H:%M:%S.%f").timetuple()))							
			if (t < min_t):
				min_t = t
			if (t > max_t):
				max_t = t	

	flag_first = False
	ct = 0	

	im = read_tomo(dset,0).astype(float32)
		
	for i in range(0, provenance_dset.shape[0]):  
		if provenance_dset["filename", i].startswith(darkorflatprefix):
			t = int(mktime(datetime.strptime(provenance_dset["timestamp", i], "%Y-%m-%d %H:%M:%S.%f").timetuple()))							

			if flagafter:
				if (t > max_t):				
				
					# Get "angle":
					name = splitext(provenance_dset["filename", i])[0]
					idx = int(name[-4:])
					idx = idx - int(provenance_dset.attrs['first_index'])
					
					# Get image and sum it to the series:
					if not flag_first:
						flag_first = True					
						im = read_tomo(dset,idx).astype(float32)
						ct = ct + 1
					else:
						im = im + read_tomo(dset,idx).astype(float32) 
						ct = ct + 1
			else:
				if (t <= min_t):					
				
					# Get "angle":
					name = splitext(provenance_dset["filename", i])[0]
					idx = int(name[-4:])
					idx = idx - int(provenance_dset.attrs['first_index'])
					
					# Get image and sum it to the series:
					if not flag_first:
						flag_first = True					
						im = read_tomo(dset,idx).astype(float32)					
					else:
						im = im + read_tomo(dset,idx).astype(float32) 
					
					ct = ct + 1
	
	
	if ( ct > 0 ):
		im = im / ct
		return im.astype(float32)  	
	
	else:		
		return -1 # Error:

def _medianize(dset):

	num_imgs = get_nr_projs ( dset )

	# Return error if there are no images:
	if (num_imgs == 0):      
		return -1 # Error:

	# Return the only image in case of one image:
	elif (num_imgs == 1):
		return read_tomo(dset,0).astype(float32)
	
	# Do the median of all the images if there is more than one image:
	if num_imgs > 1:

		# Get first image:
		im = read_tomo(dset,0).astype(float32)
		
		# Read all the remaining files (if any) and save it in a volume:
		for i in range(1, num_imgs):                 
		
			# Read i-th image from input folder:
			#im = im + dset[i,:,:].astype(float32)   
			im = im + read_tomo(dset,i).astype(float32) 
		
		# Medianize volume along the third-dimension:
		im = im / num_imgs

		# Reshape output and return:
		return im.astype(float32)  
		

def extract_flatdark(f_in, flat_end, logfilename):
	"""Extract the flat and dark reference images to be used during the pre-processing step.

    Parameters
    ----------
    f_in : HDF5 data structure
        The data structure containing the flat and dark acquired images.

    flat_end : bool
        Consider the flat/dark images acquired after the projections (if any).

	logilename : string
		Absolute file of a log text file where infos are appended.
    
    """
	skip_flat = False
	skip_flat_after = not flat_end
	flat_onlyafter = False
	
	# Prepare output variables:
	im_flat = 0
	im_dark = 0
	im_flat_after = 0
	im_dark_after = 0

	# Get prefixes:
	if "/tomo" in f_in:
		dset = f_in['tomo']
		
		tomoprefix = 'tomo'
		flatprefix = 'flat'
		darkprefix = 'dark'
	else: 
		dset = f_in['exchange/data']
		if "/provenance/detector_output" in f_in:
			prov_dset = f_in['provenance/detector_output']		
			
			tomoprefix = prov_dset.attrs['tomo_prefix']
			flatprefix = prov_dset.attrs['flat_prefix']
			darkprefix = prov_dset.attrs['dark_prefix']

	# Get dark images:
	if "/dark" in f_in:
		im_dark = _medianize(f_in['dark'])			
			
		if not isinstance(im_dark, ndarray):
			log = open(logfilename,"a")
			if not flat_end:
				skip_flat = True
				log.write(linesep + "\tNo dark field images (acquired before the projections) found. 'Use after' flag not specified. Flat fielding skipped.")	
			else:
				flat_onlyafter = True
				log.write(linesep + "\tNo dark field images (acquired before the projections) found.")		
			log.close()		
		else:
			log = open(logfilename,"a")
			log.write(linesep + "\tDark field images (acquired before the projections) found.")	
			log.close()
				
	elif "/exchange/data_dark" in f_in:
		
		# Get the dark files acquired before the projections:	
		im_dark = _medianize_withprovenance(f_in['exchange/data_dark'], f_in['provenance/detector_output'], tomoprefix, darkprefix, False)		
			
		if not isinstance(im_dark, ndarray):
			log = open(logfilename,"a")
			if not flat_end:
				skip_flat = True
				log.write(linesep + "\tNo dark field images (acquired before the projections) found. 'Use after' flag not specified. Flat fielding skipped.")	
			else:
				flat_onlyafter = True
				log.write(linesep + "\tNo dark field images (acquired before the projections) found.")	
			log.close()				
		else:
			log = open(logfilename,"a")
			log.write(linesep + "\tDark field images (acquired before the projections) found.")	
			log.close()
	else:		
		log = open(logfilename,"a")
		if not flat_end:
			skip_flat = True
			log.write(linesep + "\tNo dark field images (acquired before the projections) found. 'Use after' flag not specified. Flat fielding skipped.")	
		else:
			flat_onlyafter = True
			log.write(linesep + "\tNo dark field images (acquired before the projections) found.")	
		log.close()	
	


	# Get flat images:		
	if "/flat" in f_in:
		
		im_flat = _medianize(f_in['flat'])	
			
		if not isinstance(im_flat, ndarray):
			log = open(logfilename,"a")
			if not flat_end:
				skip_flat = True
				log.write(linesep + "\tNo flat field images (acquired before the projections) found. 'Use after' flag not specified. Flat fielding skipped.")	
			else:
				flat_onlyafter = True
				log.write(linesep + "\tNo flat field images (acquired before the projections) found.")	
		else:
			log = open(logfilename,"a")
			log.write(linesep + "\tFlat field images (acquired before the projections) found.")	
			log.close()
				
	elif "/exchange/data_white" in f_in:
		
		# Get the flat files acquired before the projections:						
		im_flat = _medianize_withprovenance(f_in['exchange/data_white'], f_in['provenance/detector_output'], tomoprefix, flatprefix, False)
			
		if not isinstance(im_flat, ndarray):
			log = open(logfilename,"a")
			if not flat_end:
				skip_flat = True
				log.write(linesep + "\tNo flat field images (acquired before the projections) found. 'Use after' flag not specified. Flat fielding skipped.")	
			else:
				flat_onlyafter = True
				log.write(linesep + "\tNo flat field images (acquired before the projections) found.")	
			log.close()	
				
		else:
			log = open(logfilename,"a")
			log.write(linesep + "\tFlat field images (acquired before the projections) found.")	
			log.close()				
	else:
		log = open(logfilename,"a")
		if not flat_end:
			skip_flat = True
			log.write(linesep + "\tNo flat field images (acquired before the projections) found. 'Use after' flag not specified. Flat fielding skipped.")	
		else:
			flat_onlyafter = True
			log.write(linesep + "\tNo flat field images (acquired before the projections) found.")		
		log.close()		



	#
	# Flats and dark at the end (if required):
	#
	
	if not skip_flat:
		if flat_end:
						
			skip_flat_after = False
			
			if "/dark_post" in f_in:
				
				im_dark_after = _medianize(f_in['dark_post'])			
					
				if not isinstance(im_dark_after, ndarray):
					log = open(logfilename,"a")
					if flat_onlyafter:
						skip_flat = True
						log.write(linesep + "\tNo dark field images (acquired after the projections) found. Flat fielding skipped. ")	
					else:
						log.write(linesep + "\tNo dark field images (acquired after the projections) found.")	
					log.close()	
					skip_flat_after = True	
				else:
					log = open(logfilename,"a")
					log.write(linesep + "\tDark field images (acquired after the projections) found.")	
					log.close()
						
			elif "/dark_after" in f_in:
				
				im_dark_after = _medianize(f_in['dark_after'])			
					
				if not isinstance(im_dark_after, ndarray):
					log = open(logfilename,"a")
					if flat_onlyafter:
						skip_flat = True
						log.write(linesep + "\tNo dark field images (acquired after the projections) found. Flat fielding skipped. ")	
					else:
						log.write(linesep + "\tNo dark field images (acquired after the projections) found.")	
					log.close()	
					skip_flat_after = True	
				else:
					log = open(logfilename,"a")
					log.write(linesep + "\tDark field images (acquired after the projections) found.")	
					log.close()
						
			elif "/exchange/data_dark" in f_in:			
			
				im_dark_after = _medianize_withprovenance(f_in['exchange/data_dark'], f_in['provenance/detector_output'], tomoprefix, darkprefix, True)		
					
				if not isinstance(im_dark_after, ndarray):
					log = open(logfilename,"a")
					if flat_onlyafter:
						skip_flat = True
						log.write(linesep + "\tNo dark field images (acquired after the projections) found. Flat fielding skipped. ")	
					else:
						log.write(linesep + "\tNo dark field images (acquired after the projections) found.")	
					log.close()	
					skip_flat_after = True	
				else:
					log = open(logfilename,"a")
					log.write(linesep + "\tDark field images (acquired after the projections) found.")	
					log.close()
						
			else:
				log = open(logfilename,"a")
				if flat_onlyafter:
					skip_flat = True
					log.write(linesep + "\tNo dark field images (acquired after the projections) found. Flat fielding skipped. ")	
				else:
					log.write(linesep + "\tNo dark field images (acquired after the projections) found.")	
				log.close()	
				skip_flat_after = True	
					
			if "/flat_post" in f_in:
				
				im_flat_after = _medianize(f_in['flat_post'])	
					
				if not isinstance(im_flat_after, ndarray):
					log = open(logfilename,"a")
					if flat_onlyafter:
						skip_flat = True
						log.write(linesep + "\tNo flat field images (acquired after the projections) found. Flat fielding skipped. ")	
					else:
						log.write(linesep + "\tNo flat field images (acquired after the projections) found.")
					log.close()	
					skip_flat_after = True
				else:
					log = open(logfilename,"a")
					log.write(linesep + "\tFlat field images (acquired after the projections) found.")	
					log.close()
						
			elif "/flat_after" in f_in:
				
				im_flat_after = _medianize(f_in['flat_after'])	
					
				if not isinstance(im_flat_after, ndarray):
					log = open(logfilename,"a")
					if flat_onlyafter:
						skip_flat = True
						log.write(linesep + "\tNo flat field images (acquired after the projections) found. Flat fielding skipped. ")	
					else:
						log.write(linesep + "\tNo flat field images (acquired after the projections) found.")
					log.close()	
					skip_flat_after = True
				else:
					log = open(logfilename,"a")
					log.write(linesep + "\tFlat field images (acquired after the projections) found.")	
					log.close()
						
			elif "/exchange/data_white" in f_in:
				
				im_flat_after = _medianize_withprovenance(f_in['exchange/data_white'], f_in['provenance/detector_output'], tomoprefix, flatprefix, True)		
					
				if not isinstance(im_flat_after, ndarray):
					log = open(logfilename,"a")
					if flat_onlyafter:
						skip_flat = True
						log.write(linesep + "\tNo flat field images (acquired after the projections) found. Flat fielding skipped. ")	
					else:
						log.write(linesep + "\tNo flat field images (acquired after the projections) found.")	
					log.close()	
					skip_flat_after = True
				else:
					log = open(logfilename,"a")
					log.write(linesep + "\tFlat field images (acquired after the projections) found.")	
					log.close()
	
			else:
				log = open(logfilename,"a")
				if flat_onlyafter:
					skip_flat = True
					log.write(linesep + "\tNo flat field images (acquired after the projections) found. Flat fielding skipped. ")	
				else:
					log.write(linesep + "\tNo flat field images (acquired after the projections) found.")
				log.close()	
				skip_flat_after = True	

	
	#log = open(logfilename,"a")
	#log.write(linesep + "\tFlat and dark images browsed correctly.")	
	#log.close()	

	return {'im_flat':im_flat, 'im_flat_after':im_flat_after, 'im_dark':im_dark, 
			'im_dark_after':im_dark_after, 'skip_flat':skip_flat, 'skip_flat_after':skip_flat_after}
