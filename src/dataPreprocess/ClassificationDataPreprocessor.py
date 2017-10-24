from dataPreprocess.DataPreprocessor import *
# from resultEvaluation.GCI import *


class ClassificationDataPreprocessor(DataPreprocessor):
    # Transform GCI locations to label(binary classification) sequence.
    def transLocations2LabelSeq(self, locations, framedLength, samplingRate):
        zero = numpy.zeros(shape=(framedLength, 1), dtype=numpy.float32)
        one = numpy.ones(shape=(framedLength, 1), dtype=numpy.float32)
        labelSeq = numpy.reshape(numpy.asarray([zero, one]).transpose(), [framedLength, 2])
        logging.debug("mark data shape:" + str(labelSeq.shape))
        for location in locations:
            labelIndex = self.getLabelIndex(location, samplingRate, framedLength)
            # logging.debug("Time:" + str(labelLocation))
            labelSeq[labelIndex][0] = 1
            labelSeq[labelIndex][1] = 0
            pass
        return labelSeq

    # def transLocations2GCIMask(self, locations, framedLength, samplingRate):
    #     frameCount = self.getFrameCount(framedLength)
    #     reference = list()
    #     for location in locations:
    #         labelIndex = self.getLabelIndex(location, samplingRate, framedLength)
    #         reference.append(labelIndex)
    #         pass
    #     gciList = transRef2GCIList(reference, self.defaultRadius)
    #     gciMaskMatrix = transGCIList2GCIMaskVector(gciList, frameCount)#  GCIMaskMatrix shape: (nGCIs, frameCount)
    #     maskMatrix = np.zeros(shape=(frameCount, frameCount), dtype=numpy.float32)
    #     for i, labelIndex in enumerate(reference):
    #         maskMatrix[labelIndex] = gciMaskMatrix[i]
    #         pass
    #     return maskMatrix

    def setDefaultRadius(self, defaultRadius):
        self.defaultRadius = defaultRadius
        pass

    pass