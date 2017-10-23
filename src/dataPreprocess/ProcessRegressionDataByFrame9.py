from dataPreprocess.RegressionDataPreprocessor import *

dataFilePath = "data/hdf5/APLAWDW_frame_9_regression.hdf5"
dataPreprocessor = RegressionDataPreprocessor(dataFilePath,
                                              frameSize=9,
                                              frameStride=5,
                                              waveDirPath="data/APLAWDW/wav/",
                                              waveExtension=".wav",
                                              markDirPath="data/APLAWDW/mark/",
                                              markExtension=".marks")
dataPreprocessor.process()
