import math

from dataAccessor.utils import pad_sequences


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
        startKeyIndex = batchIndex * self.batchSize
        endKeyIndex = (batchIndex + 1) * self.batchSize
        for i in range(startKeyIndex, endKeyIndex, 1):
            j = i % self.keyList.__len__()
            if j == 0:
                self.completedEpoch += 1
                pass
            sentenceX = self.dataFile[self.keyList[j] + "/input"]
            sentenceY = self.dataFile[self.keyList[j] + "/label"]
            batchX.append(sentenceX)
            batchY.append(sentenceY)
            pass
        batchX, _ = pad_sequences(batchX, maxlen=self.maxTimeStep)
        batchY, _ = pad_sequences(batchY, maxlen=self.maxTimeStep)
        return (batchX, batchY)

    def getBatchKeyList(self, batchIndex):
        startKeyIndex = batchIndex * self.batchSize
        endKeyIndex = (batchIndex + 1) * self.batchSize
        keyList = self.keyList[startKeyIndex:endKeyIndex]
        return keyList

    pass
