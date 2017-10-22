from dataPreprocess.DataPreprocessor import *


class ClassificationDataPreprocessor(DataPreprocessor):
    # Transform GCI locations to label(binary classification) sequence.
    def transLocations2LabelSeq(self, locations, labelSeqLength, samplingRate):
        zero = numpy.zeros(shape=(labelSeqLength, 1), dtype=numpy.float32)
        one = numpy.ones(shape=(labelSeqLength, 1), dtype=numpy.float32)
        labelSeq = numpy.reshape(numpy.asarray([zero, one]).transpose(), [labelSeqLength, 2])
        logging.debug("mark data shape:" + str(labelSeq.shape))
        for location in locations:
            labelLocation = floor(int(location * samplingRate / self.frameSize)) - 1
            # logging.debug("Time:" + str(labelLocation))
            labelSeq[labelLocation][0] = 1.0
            labelSeq[labelLocation][1] = 0.0
            pass
        return labelSeq

    pass