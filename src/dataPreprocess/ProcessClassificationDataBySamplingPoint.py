from dataPreprocess.ClassificationDataPreprocessor import *

dataFilePath = "data/hdf5/APLAWDW_sampling_point_classification.hdf5"
dataPreprocessor = DataPreprocessor(dataFilePath,
                                    frameSize=1,
                                    frameStride=1,
                                    waveDirPath="data/APLAWDW/wav/",
                                    waveExtension=".wav",
                                    markDirPath="data/APLAWDW/mark/",
                                    markExtension=".marks")
dataPreprocessor.setDefaultRadius(defaultRadius = 400)
dataPreprocessor.process()
