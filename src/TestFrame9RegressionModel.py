from dnnModel.RegressionModel import *


class TestFrame9RegressionModel(object):
    def __init__(self):
        self.model = RegressionModel(inputSize=9, timeStepSize=None, hiddenSize=256, layerNum=2, classNum=2,
                                learningRate=1e-3)
        self.dataFilename = "data/hdf5/APLAWDW_frame_9_regression.hdf5"
        self.resultFilename = "data/hdf5/APLAWDW_frame_9_regression_result_test.hdf5"
        self.modelRestorePath = "model/APLAWDW_frame_9_regression_model"
        self.modelSavePath = None
        pass

    def run(self):
        self.model.setDataFilename(self.dataFilename, self.resultFilename)
        self.model.setModelSavePath(self.modelRestorePath, self.modelSavePath)
        self.model.test(samplingRate=20000)
        pass

    pass


def main():
    frameModel = TestFrame9RegressionModel()
    frameModel.run()
    pass


if __name__ == '__main__':
    main()
    pass
