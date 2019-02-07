import sys
sys.path.insert(0, '../../../../class')

import os
import psutil
import time
import nnet
import pickle
import FFNN as ffnn
import CubeLattice as cl
from scipy.io import loadmat


if __name__ == '__main__':

    filemat = loadmat('ACASXU_run2a_1_1_batch_2000.mat')
    W = filemat['W'][0]
    b = filemat['b'][0]
    lb = filemat['Minimum_of_Inputs'][0]
    ub = filemat['Maximum_of_Inputs'][0]
    range_for_scaling = filemat['range_for_scaling'][0]
    means_for_scaling = filemat['means_for_scaling'][0]

    for i in range(5):
        lb[i] = (lb[i] - means_for_scaling[i]) / range_for_scaling[i]
        ub[i] = (ub[i] - means_for_scaling[i]) / range_for_scaling[i]


    # no memory optimization
    nnet = nnet.nnetwork(W, b, False)

    cube_lattice = cl.CubeLattice(lb, ub)
    initial_input = cube_lattice.to_poly_lattice()

    start_time = time.time()

    # no parallel
    outputSets = ffnn.nnet_output(nnet, initial_input, )

    # time elapsed
    elapsed_time = time.time() - start_time
    print("time elapsed: ", elapsed_time, 'seconds')

    # memory occupied
    process = psutil.Process(os.getpid())
    print("memory occupied: ", process.memory_info().rss/1e+9,'Gb')  # in bytes

    # number of polytopes
    print("number of polytopes: ", len(outputSets))

    with open('outputSets', 'wb') as f:
        pickle.dump([outputSets], f)

    file = open('output_info.txt', 'w')
    file.write('time elapsed: %f seconds \n' %elapsed_time)
    file.write('memory occupied: %f Gb \n' % (process.memory_info().rss/1e+9))
    file.write('number of polytopes: %d \n' % len(outputSets))
    file.close()


