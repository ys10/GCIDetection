from dnnModel.BLSTMModel import *


class TestFrameModel(object):
    def __init__(self):
        self.model = BLSTMModel(inputSize=9, timeStepSize=None, hiddenSize=256, layerNum=2, classNum=2,
                                learningRate=1e-3)
        self.dataFilename = "data/hdf5/APLAWDW_frame_9.hdf5"
        self.resultFilename = "data/hdf5/APLAWDW_frame_9_result_test.hdf5"
        pass

    def run(self):
        self.model.setDataFilename(self.dataFilename, self.resultFilename)
        self.model.test(samplingRate=20000)
        pass

    pass


def main():
    frameModel = TestFrameModel()
    frameModel.run()
    pass


if __name__ == '__main__':
    main()
    pass
