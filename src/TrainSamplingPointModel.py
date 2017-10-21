from dnnModel.BLSTMModel import *


class TrainSamplingPointModel(object):
    def __init__(self):
        self.model = BLSTMModel(inputSize=1, timeStepSize=None, hiddenSize=256, layerNum=2, classNum=2,
                                learningRate=1e-3)
        self.dataFilename = "data/hdf5/APLAWDW_sampling_point_classification.hdf5"
        self.resultFilename = "data/hdf5/APLAWDW_sampling_point_classification_result.hdf5"
        self.modelRestorePath = None
        self.modelSavePath = "model/APLAWDW_sampling_point_classification_model"
        self.summarySavePath = "summary/APLAWDW_sampling_point_classification_model/"
        pass

    def run(self):
        self.model.setDataFilename(self.dataFilename, self.resultFilename)
        self.model.setModelSavePath(self.modelRestorePath, self.modelSavePath)
        self.model.setSummarySavePath(self.summarySavePath)
        self.model.train(trainIteration=20000, saveIteration=500, displayIteration=5, batchSize=4, samplingRate=20000)
        pass

    pass


def main():
    samplingPointModel = TrainSamplingPointModel()
    samplingPointModel.run()
    pass


if __name__ == '__main__':
    main()
    pass
