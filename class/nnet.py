import numpy as np
import time
import os
import sys
import psutil
import pickle
import multiprocessing
from functools import partial
from multiprocessing import get_context


class nnetwork:

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.c = 0
        self.numLayer = len(W)
        self.output_type = 0 # "number_of_output", "output_polytopes"
        self.pool = multiprocessing.Pool(1) # multiprocessing

    # nn output of input starting from mth layer
    def layerOutput(self, inputPoly, m):
        # print('Layer: ',m)

        inputSets = [inputPoly]
        for i in range(m, self.numLayer):
            outputPolys = []
            for aPoly in inputSets:
                outputPolys.extend(self.singleLayerOutput(aPoly, i))
            inputSets = outputPolys

        return inputSets


    # point output of nn
    def outputPoint(self, inputPoint):
        for i in range(self.numLayer):
            inputPoint = self.singleLayerPointOutput(inputPoint, i)

        return inputPoint

    # point output of single layer
    def singleLayerPointOutput(self, inputPoint, layerID):
        W = self.W[layerID]
        b = self.b[layerID]
        layerPoint = np.dot(W, inputPoint.transpose())+b
        if layerID == self.numLayer-1:
            return layerPoint.transpose()
        else:
            layerPoint[layerPoint<0] = 0
            return layerPoint.transpose()


    # polytope output of single layer
    def singleLayerOutput(self, inputPoly, layerID):
        # print("layer", layerID)
        # inputPoly = shared_inputSets[inputSets_index]
        W = self.W[layerID]
        b = self.b[layerID]
        numNeuron = b.shape[0]
        inputPoly.linearTrans(W, b)

        # partition graph sets according to properties of the relu function
        if layerID == self.numLayer-1:
            if self.output_type == 0:
                inputPoly.lattice = []
                inputPoly.ref_vertex = {}
                return [inputPoly]

            if self.output_type == 1:
                return [inputPoly]

        # #
        polys = [inputPoly]
        for i in range(numNeuron):
            splited_polys = []
            for aPoly in polys:
                splited_polys.extend(self.splitPoly(aPoly, i))

            polys = splited_polys

        return polys


    # partition one input polytope with a hyberplane
    def splitPoly(self, inputPoly, idx):
        outputPolySets = []

        pPolyG, nPolyG, t = inputPoly.intersectPlane(idx)
        self.c = self.c+t

        if pPolyG:
            outputPolySets.append(pPolyG)

        if nPolyG:
            nPolyG_new = nPolyG.map_negative_poly(idx)
            outputPolySets.append(nPolyG_new)

        return outputPolySets



    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
