import numpy as np


class ResultWriter(object):
    def __init__(self, samplingRate, frameSize, frameStride):
        self.samplingRate = samplingRate
        self.frameSize = frameSize
        self.frameStride =frameStride
        pass

    def setResultFile(self, resultFile):
        self.resultFile = resultFile
        pass

    def saveBatchResult(self, batchResult, keyList):
        for i in range(keyList.__len__()):
            locations = self.transLabelSeq2Locations(batchResult[i], self.samplingRate, self.frameSize, self.frameStride)
            self.resultFile[keyList[i]] = locations
            print("key:" + str(keyList[i]))
            print("locations:" + str(locations))
            pass
        pass

    # Transform label(binary classification) sequence to GCI locations
    def transLabelSeq2Locations(self, labelSeq, samplingRate, frameSize, frameStride):
        locations = list()
        # labelLocations = np.where(np.array(labelSeq)[:, 0] == 1)[0].tolist()
        labelLocations = list(np.where(np.array(labelSeq)[:] == 1))
        for labelLocation in labelLocations:
            # location = (labelLocation + 1) * frameSize / samplingRate
            location =(labelLocation * frameStride + frameSize / 2)/ samplingRate
            locations.append(location)
            pass
        return locations

    pass


def trans2DLabelSeq2Locations(labelSeq, samplingRate, frameSize, frameStride):
    locations = list()
    labelLocations = np.where(np.array(labelSeq)[:, 0] == 1)[0].tolist()
    for labelLocation in labelLocations:
        location =(labelLocation * frameStride + frameSize / 2)/ samplingRate
        locations.append(location)
        pass
    return locations

# Transform label(binary classification) sequence (timeSteps, 2) to GCI locations
def trans1DLabelSeq2Locations(labelSeq, samplingRate=20000, frameSize=9, frameStride=9):
    locations = list()
    labelLocations = list(np.where(np.array(labelSeq)[:] == 1))
    for labelLocation in labelLocations:
        location =(labelLocation * frameStride + frameSize / 2)/ samplingRate
        locations.append(location)
        pass
    return locations
