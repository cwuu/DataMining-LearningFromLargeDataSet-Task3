###################################################################################################
##
## Data Mining: Learning from Large Datasets
## Lecture Project - Task 3 (Large Scale Clustering)
## 
## Team: 	Datasloths
## Authors: Raphael S. (rasuter@student.ethz.ch)
## 			Hwang S. (hwangse@student.ethz.ch)
## 			Wuu Cheng-Hsin (wch@student.ethz.ch)
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
runs = 2  # number of runs of k-means (choose lowest error)
tol = 0.05  # percentage of points that can change their cluster in a step before Lloyd's algo terminates

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


def weighted_kmeans(weights, vectors, k, tol):
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
		if np.sum(cluster_old != cluster) < tol*num_vecs:
			break

	sq_distance = np.amin(sq_distances_to_clusters, axis=1)
	quantization_error = np.dot(weights, sq_distance)

	return (mus, quantization_error)


## MAPPER #########################################################################################

def mapper(key, value):
	"""
	@brief      yields the value it receives
	"""

	yield 0, value  # in the first column we have the weights

## REDUCER ########################################################################################

def reducer(key, values):
	"""
	@brief      Runs Lloyd's algorithm multiple times on the full data and returns the cluster
				center which results in the lowest error
	"""

	global k, runs, tol

	vectors = values
	weights = np.ones([vectors.shape[0]]) / vectors.shape[0]

	all_mus = []
	all_errors = []
	for _ in range(runs):
		mus, quantization_error = weighted_kmeans(weights, vectors, k, tol)
		all_mus.append(mus)
		all_errors.append(quantization_error)

	print 'Quantization Errors:', all_errors

	yield all_mus[np.argmin(all_errors)]
