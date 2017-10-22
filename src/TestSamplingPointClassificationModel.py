from dnnModel.BLSTMModel import *


class TestSamplingPointClassificationModel(object):
    def __init__(self):
        self.model = BLSTMModel(inputSize=1, timeStepSize=None, hiddenSize=256, layerNum=2, classNum=2,
                                learningRate=1e-3)
        self.dataFilename = "data/hdf5/APLAWDW_sampling_point_classification.hdf5"
        self.resultFilename = "data/hdf5/APLAWDW_sampling_result.hdf5"
        self.modelRestorePath = "model/APLAWDW_sampling_point_classification_model"
        self.modelSavePath = None
        pass

    def run(self):
        self.model.setDataFilename(self.dataFilename, self.resultFilename)
        self.model.setModelSavePath(self.modelRestorePath, self.modelSavePath)
        self.model.test(samplingRate=20000)
        pass

    pass


def main():
    frameModel = TestSamplingPointClassificationModel()
    frameModel.run()
    pass


if __name__ == '__main__':
    main()
    pass
