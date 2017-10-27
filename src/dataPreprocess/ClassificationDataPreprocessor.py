from dataPreprocess.DataPreprocessor import *
import numpy
from dataAccessor.ResultWriter import *
# from resultEvaluation.GCI import *


class ClassificationDataPreprocessor(DataPreprocessor):
    # Transform GCI locations to label(binary classification) sequence.
    def transLocations2LabelSeq(self, locations, framedLength, samplingRate):
        logging.info("locations:"+ str(locations))
        logging.debug("frameLength:"+str(framedLength))
        labelSeq = numpy.zeros(shape=(framedLength,), dtype=numpy.float32)
        # zero = numpy.zeros(shape=(framedLength, 1), dtype=numpy.float32)
        # one = numpy.ones(shape=(framedLength, 1), dtype=numpy.float32)
        # labelSeq = numpy.reshape(numpy.asarray([zero, one]).transpose(), [framedLength, 2])
        # logging.debug("mark data shape:" + str(labelSeq.shape))
        for location in locations:
            labelIndex = self.getLabelIndex(location, samplingRate, framedLength)
            logging.debug("Location:" + str(location))
            logging.debug("Time:" + str(labelIndex))
            labelSeq[labelIndex] = 1.0
            # labelSeq[labelIndex][0] = 1.0
            # labelSeq[labelIndex][1] = 0.0
            pass
        # testLabelSeq = trans1DLabelSeq2Locations(labelSeq.transpose()[0], samplingRate, self.frameSize, self.frameStride)
        testLabelSeq = trans1DLabelSeq2Locations(labelSeq, samplingRate, self.frameSize, self.frameStride)
        logging.info("labelSeq:" + str(testLabelSeq))
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