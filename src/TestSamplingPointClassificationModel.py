from dnnModel.ClassificationModel import *
from dataAccessor.ClassificationResultWriter import *


class TestSamplingPointClassificationModel(object):
    def __init__(self):
        self.model = ClassificationModel(inputSize=1, timeStepSize=None, hiddenSize=256, layerNum=2, classNum=2,
                                         learningRate=1e-3)
        self.samplingRate = 20000
        self.frameSize = 1
        self.dataFilename = "data/hdf5/APLAWDW_sampling_point_classification.hdf5"
        self.resultFilename = "data/hdf5/APLAWDW_sampling_result.hdf5"
        self.modelRestorePath = "model/APLAWDW_sampling_point_classification_model"
        self.modelSavePath = None
        self.resultWriter = ClassificationResultWriter(self.samplingRate, self.frameSize)
        pass

    def run(self):
        self.model.setDataFilename(self.dataFilename, self.resultFilename)
        self.model.setModelSavePath(self.modelRestorePath, self.modelSavePath)
        self.model.test(self.samplingRate)
        pass

    pass


def main():
    frameModel = TestSamplingPointClassificationModel()
    frameModel.run()
    pass


if __name__ == '__main__':
    main()
    pass
