from dnnModel.BLSTMModel import *


class TrainSamplingPointModel(object):
    def __init__(self):
        self.model = BLSTMModel(inputSize=1, timeStepSize=None, hiddenSize=256, layerNum=2, classNum=2,
                                learningRate=1e-3)
        self.dataFilename = "data/hdf5/APLAWDW_s_01_a.hdf5"
        self.resultFilename = "data/hdf5/APLAWDW_sampling_result.hdf5"
        self.summarySavePath = "summary/"
        pass

    def run(self):
        self.model.setDataFilename(self.dataFilename, self.resultFilename)
        self.model.setSummarySavePath(self.summarySavePath)
        self.model.train(trainIteration=2, saveIteration=1, displayIteration=1, batchSize=1, samplingRate=20000)
        pass

    pass


def main():
    samplingPointModel = TrainSamplingPointModel()
    samplingPointModel.run()
    pass


if __name__ == '__main__':
    main()
    pass
