import os
import sys
import multiprocessing
from functools import partial

# nnet: neural network
# inputPoly: polytope input
# method: parallel or non-parallel
# method = (parallel, number_of_cores) or (non_parallel, )
def nnet_output(nnet, inputPoly, method=("non_parallel",)):
    print("running...")
    if method[0] is not "parallel":
        outputSets = nnet.layerOutput(inputPoly, 0)
        return outputSets
    else:
        local_cores = os.cpu_count()
        if method[1]>local_cores:
            print("The number of local cores is", local_cores)
            print("The selected number of cores is too large")
            sys.exit(1)

        nputSets0 = nnet.singleLayerOutput(inputPoly, 0)
        outputSets = []
        pool = multiprocessing.Pool(method[1])
        outputSets.extend(pool.map(partial(nnet.layerOutput, m=1), nputSets0))
        outputSets = [item for sublist in outputSets for item in sublist]
        return outputSets



