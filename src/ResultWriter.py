import numpy as np


class ResultWriter(object):
    def __init__(self, resultFile, samplingRate):
        self.resultFile = resultFile
        self.samplingRate = samplingRate
        pass

    def saveBatchResult(self, batchResult, keyList):
        for i in range(keyList.__len__()):
            self.resultFile[keyList[i]] = self.transLabelSeq2Locations(batchResult[i], self.samplingRate)
            pass
        pass

    # Transform label(binary classification) sequence to GCI locations
    def transLabelSeq2Locations(self, labelSeq, samplingRate):
        locations = list()
        labelLocations = np.where(np.array(labelSeq)[:, 0] == 1)[0].tolist()
        for labelLocation in labelLocations:
            location = (labelLocation + 1) / samplingRate
            locations.append(location)
            pass
        return locations

    pass
