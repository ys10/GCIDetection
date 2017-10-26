from dataPreprocess.ClassificationDataPreprocessor import *

dataFilePath = "data/hdf5/APLAWDW_frame_9_classification_validation.hdf5"
dataPreprocessor = ClassificationDataPreprocessor(dataFilePath,
                                                  frameSize=9,
                                                  frameStride=9,
                                                  waveDirPath="data/APLAWDW/wav/",
                                                  waveExtension=".wav",
                                                  markDirPath="data/APLAWDW/valid_mark/",
                                                  markExtension=".marks")
dataPreprocessor.setDefaultRadius(defaultRadius = 40)
dataPreprocessor.process()
