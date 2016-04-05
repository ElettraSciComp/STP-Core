from numpy import float32, linspace

import tomopy

def recon_tomopy_iterative(im, angles, method, iterations):	

	rec = zeros((im.shape[1],  im.shape[1]))
	rec = rec + 1.0
	theta = linspace(0, angles, im.shape[0], endpoint=False).astype(float32)		

	im = im * im.shape[1]
	rec_im = tomopy.mlem(im.astype(float32), theta, im.shape[1], int(iterations), rec.astype(float32))	
	
	return rec_im.astype(float32)