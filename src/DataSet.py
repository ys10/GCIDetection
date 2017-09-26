import numpy
import math
from utils import pad_sequences

class DataSet(object):

    def __init__(self, dataFile, batchSize, maxTimeStep):
        self.dataFile = dataFile
        self.batchSize = batchSize
        self.maxTimeStep = maxTimeStep
        self.keyList = list(dataFile.keys())
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
            batchX, _ = pad_sequences(batchX, maxlen=self.maxTimeStep)
            # batchX = numpy.fromstring(batchX, dtype=numpy.float32)
            batchY, _ = pad_sequences(batchY, maxlen=self.maxTimeStep)
            return (batchX, batchY)
pass