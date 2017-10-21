from dnnModel.BLSTMModel import *


class TrainFrameModel(object):
    def __init__(self):
        self.model = BLSTMModel(inputSize=9, timeStepSize=None, hiddenSize=256, layerNum=2, classNum=2,
                                learningRate=1e-3)
        self.dataFilename = "data/hdf5/APLAWDW_frame_9.hdf5"
        self.resultFilename = "data/hdf5/APLAWDW_frame_9_result.hdf5"
        self.summarySavePath = "summary/"
        pass

    def run(self):
        self.model.setDataFilename(self.dataFilename, self.resultFilename)
        self.model.setSummarySavePath(self.summarySavePath)
        self.model.train(trainIteration=200, saveIteration=100, displayIteration=5, batchSize=16, samplingRate=16000)
        pass

    pass


def main():
    frameModel = TrainFrameModel()
    frameModel.run()
    pass


if __name__ == '__main__':
    main()
    pass
