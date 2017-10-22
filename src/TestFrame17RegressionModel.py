from dnnModel.RegressionModel import *
from dataAccessor.RegressionResultWriter import *


class TestFrame17RegressionModel(object):
    def __init__(self):
        self.model = RegressionModel(inputSize=17, timeStepSize=None, hiddenSize=256, layerNum=2, classNum=2,
                                learningRate=1e-3)
        self.samplingRate = 20000
        self.frameSize = 17
        self.dataFilename = "data/hdf5/APLAWDW_frame_17_regression.hdf5"
        self.resultFilename = "data/hdf5/APLAWDW_frame_17_regression_result_test.hdf5"
        self.modelRestorePath = "model/APLAWDW_frame_17_regression_model"
        self.modelSavePath = None
        self.resultWriter = RegressionResultWriter(self.samplingRate, self.frameSize)
        pass

    def run(self):
        self.model.setDataFilename(self.dataFilename, self.resultFilename)
        self.model.setModelSavePath(self.modelRestorePath, self.modelSavePath)
        self.model.test(self.samplingRate)
        pass

    pass


def main():
    frameModel = TestFrame17RegressionModel()
    frameModel.run()
    pass


if __name__ == '__main__':
    main()
    pass
