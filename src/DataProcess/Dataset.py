import h5py
import math
from src.DataProcess.utils import *

class Data(object):

    def __init__(self, dataFile, batchSize, timeStep):
        self.dataFile = dataFile
        self.batchSize = batchSize
        self.timeStep = timeStep
        self.keyList = dataFile.keys()
        self.batchCount = math.ceil( self.keyList.__len__() /  float(self.batchSize))
        self.completedEpoch = 0
        pass

    def getBatchCount(self):
        return self.batchCount

    def getBatch(self, batchIndex):
        if batchIndex >= self.batchCount:
            return []
        else:
            batchX = []
            batchY = []
            startKeyIndex = batchIndex * self.batchSize
            endKeyIndex = (batchIndex+ 1) * self.batchSize
            for i in range(startKeyIndex, endKeyIndex, 1) :
                j = i % self.keyList.__len__()
                if i == self.keyList.__len__():
                    self.completedEpoch += 1
                    pass
                sentenceX = self.dataFile[self.keyList[j] + "/input"]
                sentenceY = self.dataFile[self.keyList[j] + "/label"]
                batchX.append(sentenceX)
                batchY.append(sentenceY)
                pass
            batchX = pad_sequences(batchX, maxlen=self.timeStep)
            batchY = pad_sequences(batchY, maxlen=self.timeStep)
            return {'batchX' : batchX, 'batchY' : batchY}
pass