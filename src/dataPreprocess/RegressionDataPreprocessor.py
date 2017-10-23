from dataPreprocess.DataPreprocessor import *


class RegressionDataPreprocessor(DataPreprocessor):
    def __init__(self, dataFilePath, frameSize=1, frameStride=1, waveDirPath="data/wav/", waveExtension=".wav",
                 markDirPath="data/mark/",
                 markExtension=".mark"):
        DataPreprocessor.__init__(self, dataFilePath, frameSize,  frameStride, waveDirPath, waveExtension, markDirPath, markExtension)
        self.gciCriticalDistance = None
        pass

    def setGCICriticalDistance(self, gciCriticalDistance=800):
        self.gciCriticalDistance = gciCriticalDistance
        pass

    def getGCICriticalDistance(self):
        if self.gciCriticalDistance is None:
            self.gciCriticalDistance = 800
            pass
        return self.gciCriticalDistance

    def getAmendDistance(self, distance):
        if distance > self.getGCICriticalDistance():
            distance = self.getGCICriticalDistance()
            pass
        return distance

    # Transform GCI locations to label(binary classification) sequence.
    def transLocations2LabelSeq(self, locations, labelSeqLength, samplingRate):
        forward = numpy.zeros(shape=(labelSeqLength, 1), dtype=numpy.float32)
        backward = numpy.zeros(shape=(labelSeqLength, 1), dtype=numpy.float32)
        labelSeq = numpy.reshape(numpy.asarray([forward, backward]).transpose(), [labelSeqLength, 2])
        logging.debug("mark data shape:" + str(labelSeq.shape))
        labelLocations = list()
        for location in locations:
            labelLocation = floor(int(location * samplingRate / self.frameSize)) - 1
            # logging.debug("Time:" + str(labelLocation))
            labelLocations.append(labelLocation)
            pass
        for i in range(labelLocations.__len__()):
            currentLocation = labelLocations[i]
            labelSeq[currentLocation][0] = 0
            labelSeq[currentLocation][1] = 0
            # Do with the first GCI
            if i == 0:
                for j in range(currentLocation):
                    labelSeq[j][0] = self.getGCICriticalDistance()
                    labelSeq[j][1] = self.getAmendDistance(currentLocation - j)
                    pass
                pass
            # Do with the last GCI
            if i == labelLocations.__len__() - 1:
                for j in range(currentLocation + 1, labelSeq.__len__()):
                    labelSeq[j][0] = self.getAmendDistance(j - currentLocation)
                    labelSeq[j][1] = self.getGCICriticalDistance()
                    pass
                pass
            # Other location
            else:
                nextLocation = labelLocations[i + 1]
                for j in range(currentLocation + 1, nextLocation):
                    labelSeq[j][0] = self.getAmendDistance(j - currentLocation)
                    labelSeq[j][1] = self.getAmendDistance(nextLocation - j)
                    pass
                pass
            pass
        print("labelSeq:"+str(labelSeq))
        return labelSeq

    def transLocations2GCIMask(self, locations, samplingRate):
        return None

    pass


def main():
    dataFilePath = "data/hdf5/APLAWDW_test.hdf5"
    dataPreprocessor = RegressionDataPreprocessor(dataFilePath, frameSize=1)
    dataPreprocessor.process()
    pass


if __name__ == '__main__':
    main()
    pass
