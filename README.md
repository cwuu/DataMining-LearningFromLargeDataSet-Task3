# DataMining-LearningFromLargeDataSet-Task3

In this task, our original implementation is as following: (1) In Mapper, use proper corsets to represent the data points. (2) In Reducer, perform the weighted k-mean algorithm to find the center for each cluster. More specifically, we use practical corset construction to parallel computation of corsets in the mapper. D^2 sampling and importance sampling are applied to find the proper distribution q, which is used to give the weighted samples along with all the space. In Reducer, we merge and compress all the corsets to a large one and run Lloyd’s algorithm on final Corset. However, this method is not efficient enough to reach the hard baseline within the time constraint. We then applied some tricks to speed up our program, such as initializing half of the cluster’s centers randomly and applying tolerance for convergence checking in Lloyd’s algorithm. The most efficient way to improve our result is to increase the run of the k-mean algorithm, but this also increases the running time drastically. Our final solution then (score 24.3 within 5 minutes) completely discarded using corsets but just run Lloyd’s algorithm with tolerance proportional to the size of data sets to all the data points. We find the low efficiency is because we have too many cluster centers but only little data. Theoretically, it was set that one would need O(K^3) vectors in a corset for it to be efficient, but we don't have enough data. 
