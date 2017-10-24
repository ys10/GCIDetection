from dataPreprocess.ClassificationDataPreprocessor import *

dataFilePath = "data/hdf5/APLAWDW_frame_17_classification_training.hdf5"
dataPreprocessor = ClassificationDataPreprocessor(dataFilePath,
                                                  frameSize=17,
                                                  frameStride=9,
                                                  waveDirPath="data/APLAWDW/wav/",
                                                  waveExtension=".wav",
                                                  markDirPath="data/APLAWDW/mark/",
                                                  markExtension=".marks")
dataPreprocessor.setDefaultRadius(defaultRadius = 40)
dataPreprocessor.process()
