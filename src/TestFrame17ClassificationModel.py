from dnnModel.BLSTMModel import *
from dataAccessor.ClassificationResultWriter import *

class TestFrame17ClassificationModel(object):
    def __init__(self):
        self.model = BLSTMModel(inputSize=17, timeStepSize=None, hiddenSize=256, layerNum=2, classNum=2,
                                learningRate=1e-3)
        self.samplingRate = 20000
        self.frameSize = 17
        self.dataFilename = "data/hdf5/APLAWDW_frame_17_classification.hdf5"
        self.resultFilename = "data/hdf5/APLAWDW_frame_17_classification_result_test.hdf5"
        self.modelRestorePath = "model/APLAWDW_frame_17_classification_model"
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
    frameModel = TestFrame17ClassificationModel()
    frameModel.run()
    pass


if __name__ == '__main__':
    main()
    pass
