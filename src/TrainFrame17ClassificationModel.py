from  dnnModel.ClassificationModel import *

class TrainFrame17ClassificationModel(object):
    def __init__(self):
        self.model = ClassificationModel(inputSize=17, timeStepSize=None, hiddenSize=256, layerNum=2, outputSize=2, classNum=2,
                                         learningRate=2e-3)
        self.trainingDataFilename = "data/hdf5/APLAWDW_frame_17_classification_training.hdf5"
        self.validationFilename = "data/hdf5/APLAWDW_frame_17_classification_validation.hdf5"
        self.modelRestorePath = None
        self.modelSavePath = "model/APLAWDW_frame_17_classification_model"
        self.summarySavePath = "summary/APLAWDW_frame_17_classification_model/"
        pass

    def run(self):
        self.model.setTrainingDataFilename(self.trainingDataFilename, self.validationFilename)
        self.model.setModelSavePath(self.modelRestorePath, self.modelSavePath)
        self.model.setSummarySavePath(self.summarySavePath)
        self.model.setDataFileInfo(samplingRate=20000, frameSize=17, frameStride=9)
        self.model.train(trainIteration=100, saveIteration=2, displayIteration=1, batchSize=1)
        pass

    pass


def main():
    frameModel = TrainFrame17ClassificationModel()
    frameModel.run()
    pass


if __name__ == '__main__':
    main()
    pass
