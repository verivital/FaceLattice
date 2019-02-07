import sys
sys.path.insert(0, '../../class')

import os
import psutil
import time
import nnet
import pickle
import FFNN as ffnn
import CubeLattice as cl
from scipy.io import loadmat


if __name__ == '__main__':

    filemat = loadmat('NeuralNetwork7_3.mat')
    W = filemat['W'][0]
    b = filemat['b'][0]
    lb = [-1,-1,-1]
    ub = [1,1,1]

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


