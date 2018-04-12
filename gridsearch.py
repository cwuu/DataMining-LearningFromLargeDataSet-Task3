## Do Cross-Validation
## 
## 1) set list parameters to test
## 2) make sure that the parameters in the coresets_1.py script are read from the .npy files
## 2) run in shell: python2.7 gridsearch.py
## 
import os
import numpy as np


coreset_sizes = [1000]
coreset_sizes_final = [3000]
size_Bs = [200]
runss = [1,5,10,15]

#TODO be careful which script is called (1 or 2)
call = 'python2.7 runner.py data/handout_train.npy data/handout_test.npy coresets_2.py'

for coreset_size in coreset_sizes:
	for coreset_size_final in coreset_sizes_final:
	    for size_B in size_Bs:
	    	for runs in runss:
		    	alphas = [np.log2(size_B) + 1]
		    	for alpha in alphas:
					np.save('coreset_size.npy', coreset_size)
					np.save('coreset_size_final.npy', coreset_size_final)
					np.save('size_B.npy', size_B)
					np.save('runs.npy', runs)
					np.save('alpha.npy', alpha)

					print 'coreset_size', 'coreset_size_final' 'size_B', 'runs', 'alpha'
					print coreset_size, coreset_size_final, size_B, runs, alpha
					os.system(call)