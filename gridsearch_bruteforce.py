## Do Cross-Validation
## 
## 1) set list parameters to test
## 2) make sure that the parameters in the coresets_1.py script are read from the .npy files
## 2) run in shell: python2.7 gridsearch.py
## 
import os
import numpy as np
import time

runss = [1]

#TODO be careful which script is called (1 or 2)
call = 'python2.7 runner.py data/handout_train.npy data/handout_test.npy brute_force_1.py'


# 	    	for runs in runss:

# 					np.save('coreset_size.npy', coreset_size)
# 					np.save('coreset_size_final.npy', coreset_size_final)
# 					np.save('size_B.npy', size_B)
# 					np.save('runs.npy', runs)
# 					np.save('alpha.npy', alpha)

# 					print 'coreset_size', 'coreset_size_final' 'size_B', 'runs', 'alpha'
# 					print coreset_size, coreset_size_final, size_B, runs, alpha
# 					os.system(call)

for i in range(10):
	start = time.time()
	os.system(call)
	end = time.time()
	print 'Time: ', end-start, ' seconds'
