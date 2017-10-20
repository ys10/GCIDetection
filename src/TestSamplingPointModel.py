from src.DNNModel import *

class TestSamplingPointModel(object):
    def __init__(self):
        self.model = DNNModel(inputSize=1, timeStepSize=None, hiddenSize=256, layerNum=2, classNum=2, learningRate=1e-3)
        self.dataFilename = "data/hdf5/APLAWDW_sampling.hdf5"
        self.resultFilename = "data/hdf5/APLAWDW_sampling_result.hdf5"
        pass

    def run(self):
        self.model.setDataFilename(self.dataFilename, self.resultFilename)
        self.model.test(samplingRate=16000)
        pass

    pass


def main():
    frameModel = TestSamplingPointModel()
    frameModel.run()
    pass


if __name__ == '__main__':
    main()
    pass