import numpy as np


class nnetwork:

    def __init__(self, W, b, mr= "normal"):
        self.W = W
        self.b = b
        self.numLayer = len(W)
        self.memory_reduce = mr

    # nn output of input starting from mth layer
    def layerOutput(self, inputSets, m):
        for i in range(m, self.numLayer):
            inputSets = self.singleLayerOutput(inputSets, i)
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

    # polytope-set output of single layer
    def singleLayerOutput(self, inputGraphSets, layerID):
        # print("layer", layerID)
        if type(inputGraphSets) != list:
            inputGraphSets = [inputGraphSets]
        W = self.W[layerID]
        b = self.b[layerID]
        numNeuron = b.shape[0]
        layerGraphSets = [inputGraph.linearTrans(W, b) for inputGraph in inputGraphSets]

        # partition graph sets according to properties of the relu function
        if layerID == self.numLayer-1:
            return layerGraphSets
        else:
            for i in range(numNeuron):
                # print('i',i)
                layerGraphSets = self.splitPoly(layerGraphSets, numNeuron, i)

            return layerGraphSets

    # partition the input set of graphs with planes
    def splitPoly(self, inputGraphSets,numNeuron, idx):
        outputPolySets = []
        for inputGraph in inputGraphSets:

            pPolyG, nPolyG = inputGraph.intersectPlane(idx)

            if pPolyG:
                outputPolySets.append(pPolyG)

            if nPolyG:
                nPolyG_new = nPolyG.map_negative_poly(idx, self.memory_reduce)
                outputPolySets.append(nPolyG_new)

        return outputPolySets



