import math
from resultEvaluation.GCI import *
from dataAccessor.utils import *


class InputReader(object):
    def __init__(self, dataFile, batchSize, maxTimeStep):
        self.dataFile = dataFile
        self.batchSize = batchSize
        self.maxTimeStep = maxTimeStep
        self.keyList = list(dataFile.keys())
        self.batchCount = math.ceil(self.keyList.__len__() / float(self.batchSize))
        self.completedEpoch = 0
        pass

    def getBatchCount(self):
        return self.batchCount

    def getBatch(self, batchIndex):
        batchX = []
        batchY = []
        batchMask = []
        batchGCICount = []
        startKeyIndex = batchIndex * self.batchSize
        endKeyIndex = (batchIndex + 1) * self.batchSize
        for i in range(startKeyIndex, endKeyIndex, 1):
            j = i % self.keyList.__len__()
            if j == 0:
                self.completedEpoch += 1
                pass
            sentenceX = self.dataFile[self.keyList[j] + "/input"]
            sentenceY = self.dataFile[self.keyList[j] + "/label"]
            # maskMatrix = self.dataFile[self.keyList[j] + "/mask"]
            gciCount = self.dataFile[self.keyList[j] + "/gciCount"]
            batchX.append(sentenceX)
            batchY.append(sentenceY)
            # maskMatrix = self.getMaskMatrix(sentenceY)
            # batchMask.append(maskMatrix)
            maskVector = self.getMaskVector(sentenceY)
            batchMask.append(maskVector)
            batchGCICount.append(gciCount)
            pass
        batchX, _ = pad_sequences(batchX, maxlen=self.maxTimeStep)
        batchY, _ = pad_sequences(batchY, maxlen=self.maxTimeStep)
        batchMask, _ = pad_sequences(batchMask, maxlen=self.maxTimeStep)
        # batchMask = pad_matrices_dim1(pad_matrices_dim2(batchMask, maxlen=self.maxTimeStep))
        '''process y'''
        processedBatchY = []
        for y in batchY:
            one = np.ones(shape=(len(y)), dtype=np.float32)
            re_y = one - y
            y = np.reshape(np.asarray([y, re_y]).transpose(), [len(one), 2])
            processedBatchY.append(y)
            pass
        return (batchX, processedBatchY, batchMask, batchGCICount)

    def getBatchKeyList(self, batchIndex):
        startKeyIndex = batchIndex * self.batchSize
        endKeyIndex = (batchIndex + 1) * self.batchSize
        keyList = self.keyList[startKeyIndex:endKeyIndex]
        return keyList

    def getMaskMatrix(self, sentenceY, defaultRadius=1):
        maskMatrix = transSentenceY2MaskMatrix(sentenceY, defaultRadius)
        return maskMatrix

    def getMaskVector(self, sentenceY, defaultRadius=1):
        maskVector = transSentenceY2MaskVector(sentenceY, defaultRadius)
        return maskVector

    pass
