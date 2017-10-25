class TestFrame17ClassificationModel(object):
    def __init__(self):
        self.model = ClassificationModel(inputSize=17, timeStepSize=None, hiddenSize=256, layerNum=2, outputSize=2, classNum=2,
                                         learningRate=1e-3)
        self.samplingRate = 20000
        self.frameSize = 17
        self.frameStride = 9
        self.testingDataFilename = "data/hdf5/APLAWDW_frame_17_classification_training.hdf5"
        self.resultFilename = "data/hdf5/APLAWDW_frame_17_classification_result_test.hdf5"
        self.modelRestorePath = "model/APLAWDW_frame_17_classification_model-5"
        self.modelSavePath = None
        self.summarySavePath = "summary/APLAWDW_frame_17_classification_test_model/"
        self.resultWriter = ClassificationResultWriter(self.samplingRate, self.frameSize, self.frameStride)
        pass

    def run(self):
        self.model.setTestingDataFilename(self.testingDataFilename, self.resultFilename)
        self.model.openTestingDataFile()
        self.model.setModelSavePath(self.modelRestorePath, self.modelSavePath)
        self.model.setSummarySavePath(self.summarySavePath)
        self.model.setResultWriter(self.resultWriter)
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
