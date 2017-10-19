from src.DNNModel import *


class FrameModel(object):
    def __init__(self):
        self.model = DNNModel(inputSize=9, timeStepSize=None, hiddenSize=256, layerNum=2, classNum=2, learningRate=1e-3)
        self.dataFilename = "data/hdf5/APLAWDW_frame_9.hdf5"
        self.resultFilename = "data/hdf5/APLAWDW_frame_9_result.hdf5"
        pass

    def run(self):
        self.model.setDataFilename(self.dataFilename, self.resultFilename)
        self.model.train(trainIteration=200, saveIteration=100, displayIteration=5, batchSize=16)
        pass

    pass


def main():
    frameModel = FrameModel()
    frameModel.run()
    pass


if __name__ == '__main__':
    main()
    pass
