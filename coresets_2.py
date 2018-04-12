###################################################################################################
##
## Data Mining: Learning from Large Datasets
## Lecture Project - Task 3 (Large Scale Clustering)
## 
## Team: 	Datasloths
## Authors: Raphael S. (rasuter@student.ethz.ch)
## 			Hwang S. (hwangse@student.ethz.ch)
## 			.... (wch@student.ethz.ch)
## 	
## Approach:			
## 			1) Parallel Computation of Coresets in the mapper
## 			   following 'Practical coreset construction' from the lecture
## 			2) Merge all Coresets to a large one in the reducer
## 			3) (optionally: Compress large Coreset to a smaller one)
## 			4) Run Lloyd's algorithm on final Coreset
## 			
###################################################################################################

import numpy as np

## SETTINGS #######################################################################################

k = 200  # number of clusters, given in task description to be 200
# runs = 5  # number of runs of k-means (choose lowest error)
# coreset_size = 1500  # size of coreset in each mapper
# coreset_size_final = 1000  # size of the final coreset in the reducer on which k-means is run
# size_B = 200  # number of clusters considered in the Coreset construction (for D^2 sampling)
# alpha = np.log2(size_B) + 1  # parameter for importance sampling (value suggestion from old slides)

## For run of gridsearch file (read parameters from .npy files), uncomment these lines for submission
runs = int(np.load('runs.npy'))
coreset_size = int(np.load('coreset_size.npy'))
coreset_size_final = int(np.load('coreset_size_final.npy'))
size_B = int(np.load('size_B.npy'))
alpha = float(np.load('alpha.npy'))

## HELPER FUNCTIONS ###############################################################################

def d_squared_sampling(vectors, size_B, apriori_weights=1):
	"""
	@brief      Perform D^2 Sampling (as described in lecture 11, slide 29)

	@param      vectors   	original vectors, 2D numpy array of dimension (num_vecs, feature_dim)
	@param      size_B  	2D numpy array of dimension (num_vecs, feature_dim)

	@return     pair of 
					- B indizes of chosen points
					- sq_distance_to_B (2D numpy array where entry i,j stands for squared distance of vector_i to b_j)
	"""  
	(num_vecs, dim) = vectors.shape

	if apriori_weights==1:
		apriori_weights = np.ones([vectors.shape[0]]) / vectors.shape[0]
	else:
		apriori_weights /= np.sum(apriori_weights)

	sq_distance_to_B = np.inf * np.ones([num_vecs, size_B])  # distance_to_B(i,j) == squared L2-norm from point x_i to cluster center b_j
	B = size_B * [None]
	#B[0] = np.random.randint(0,num_vecs)  # start with random sample, augment list up to size size_b
	B[0] = np.random.choice(num_vecs, p=apriori_weights)
	sq_distance_to_B[:,0] = np.sum( np.square(vectors - vectors[B[0],:]), axis=1)
	for i in range(1, size_B):
		# sample according to squared distance to B
		d_squared = np.amin(sq_distance_to_B, axis=1)
		d_squared = np.multiply(d_squared, apriori_weights)  # adapt for a priori weights
		d_squared /= np.sum(d_squared)
		B[i] = np.random.choice(num_vecs, p=d_squared)  # draw samples with weighting
		sq_distance_to_B[:,i] = np.sum( np.square(vectors - vectors[B[i],:]), axis=1)  # update distances

	return B, sq_distance_to_B

def coreset_construction(value, size_B, alpha, coreset_size, apriori_weights=1):
	"""
	perform coreset construction from 2D numpy array value (vectors in rows) using D^2 sampling
	value can potentially already be a coreset, in which case the apriori weight gives the weights
	of these vectors within this coreset
	"""
	(num_vecs, dim) = value.shape

	## D^2 Sampling (lecture 11, slide 26)
	B, sq_distance_to_B = d_squared_sampling(value, size_B, apriori_weights=apriori_weights)

	## Importance Sampling (lecture 11, slide 29)
	d_squared = np.amin(sq_distance_to_B, axis=1)
	d_squared /= np.sum(d_squared)
	c_phi = np.mean(d_squared)

	q = d_squared / c_phi  # individual contribution to weighting q                       

	# compute clusterwise contribution 
	cluster = np.argmin(sq_distance_to_B, axis=1)  # which point in B is the closest
	for b in range(len(B)):
		size_Bi = np.sum(cluster==b)
		q[cluster==b] += 2*alpha * np.sum(d_squared[cluster==b]) / size_Bi / c_phi + 4*num_vecs/size_Bi

	q = np.multiply(q, apriori_weights)  # take into consideration weights of outer coreset

	q /= np.sum(q)  # normalize

	# draw samples according to q
	sample_indizes = np.random.choice(num_vecs, size=coreset_size, p=q)  # draw samples with weighting

	samples = value[sample_indizes,:]
	weights = np.reciprocal(q[sample_indizes])

	return samples, weights


def weighted_kmeans(weights, vectors, k):
	"""
    @brief      Applies Lloyds algorithm using weighted vectors
    
    @param      weights   	weights, numpy array of dimension num_vecs
    @param      vectors  	2D numpy array of dimension (num_vecs, feature_dim)
    
    @return     pair of 
    				- k clusters centern in a 2D numpy array of dim (k, feature_dim)
    				- quantization error
    """  
	
	(num_vecs, dim) = vectors.shape

	# Initialize from randomly chosen points
	# mus = vectors[np.random.choice(num_vecs, size=k, replace=False) ,:]  # initialize smarter (e.g. d^2-sampling)
	sample_idx, _ = d_squared_sampling(vectors, k)
	mus = vectors[sample_idx, :]

	sq_distances_to_clusters = np.empty([num_vecs, k])
	cluster = np.zeros(num_vecs)
	while True:
		# Assign points to closest cluster
		for i in range(k):
			sq_distances_to_clusters[:,i] = np.sum( np.square(vectors - mus[i,:]), axis=1)
		cluster_old = cluster
		cluster = np.argmin(sq_distances_to_clusters, axis=1) 

		# Compute new means
		for i in range(k):
			if len(weights[cluster==i]) > 0:  # potentially start new k-means method
				mus[i,:] = np.average(vectors[cluster==i,:], axis=0, weights=weights[cluster==i])  # compute weighted average
			else:
				return (mus, np.inf)  # have a lonely cluster center (no corresponding vectors) -> end run

		# Check for convergence
		if np.sum(cluster_old != cluster) == 0:
			break

	sq_distance = np.amin(sq_distances_to_clusters, axis=1)
	quantization_error = np.dot(weights, sq_distance)

	return (mus, quantization_error)


## MAPPER #########################################################################################

def mapper(key, value):
	"""
	@brief      Computes Coreset of vectors given in value

	@param      key    None
	@param      value  2D numpy array (3000 x 250)

	@yield     	dummy key, value = (weight, vector in Coreset)
	"""    

	global k, coreset_size, size_B, alpha

	samples, weights = coreset_construction(value, size_B, alpha, coreset_size_final)

	yield 0, np.c_[weights, samples]  # in the first column we have the weights

## REDUCER ########################################################################################

def reducer(key, values):
	"""
	@brief      Merges all collected Coresets from the mappers and runs Lloyd's algorithm on the 
				final Coreset (weighted vectors!)
				TODO: potentially compress large Coreset again before running Lloyd's algorithm

	@param      key     dummy
	@param      values  2D numpy array with columns corresponding to the vectors in the coreset
						the first column contains the weight of each vector

	@yield      200x250 numpy array containing the k-means solution vectors as columns (no key)
	"""

	global k, runs, size_B, alpha, coreset_size_final

	weights = values[:,0]
	vectors = values[:,1:]

	# construct another, smaller coreset out of this large coreset
	vectors, weights = coreset_construction(vectors, size_B, alpha, coreset_size, apriori_weights=1)

	all_mus = []
	all_errors = []
	for _ in range(runs):
		mus, quantization_error = weighted_kmeans(weights, vectors, k)
		all_mus.append(mus)
		all_errors.append(quantization_error)

	print 'Quantization Errors:', all_errors

	yield all_mus[np.argmin(all_errors)]
