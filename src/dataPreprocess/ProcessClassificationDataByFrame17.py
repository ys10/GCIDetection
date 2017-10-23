from dataPreprocess.ClassificationDataPreprocessor import *

dataFilePath = "data/hdf5/APLAWDW_frame_17_classification.hdf5"
dataPreprocessor = ClassificationDataPreprocessor(dataFilePath,
                                                  frameSize=17,
                                                  frameStride=9,
                                                  waveDirPath="data/wav/",
                                                  waveExtension=".wav",
                                                  markDirPath="data/mark/",
                                                  markExtension=".mark")
dataPreprocessor.setDefaultRadius(defaultRadius = 40)
dataPreprocessor.process()
