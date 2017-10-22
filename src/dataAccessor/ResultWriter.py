import numpy as np


class ResultWriter(object):
    def __init__(self, samplingRate, frameSize):
        self.samplingRate = samplingRate
        self.frameSize = frameSize
        pass

    def setResultFile(self, resultFile):
        self.resultFile = resultFile
        pass

    def saveBatchResult(self, batchResult, keyList):
        for i in range(keyList.__len__()):
            self.resultFile[keyList[i]] = self.transLabelSeq2Locations(batchResult[i], self.samplingRate, self.frameSize)
            pass
        pass

    # Transform label(binary classification) sequence to GCI locations
    def transLabelSeq2Locations(self, labelSeq, samplingRate, frameSize):
        locations = list()
        labelLocations = np.where(np.array(labelSeq)[:, 0] == 1)[0].tolist()
        for labelLocation in labelLocations:
            location = (labelLocation + 1) * frameSize / samplingRate
            locations.append(location)
            pass
        return locations

    pass
