from dataPreprocess.ClassificationDataPreprocessor import *

dataFilePath = "data/hdf5/APLAWDW_frame_17_classification_validation.hdf5"
dataPreprocessor = ClassificationDataPreprocessor(dataFilePath,
                                                  frameSize=17,
                                                  frameStride=9,
                                                  waveDirPath="data/APLAWDW/valid_wav/",
                                                  waveExtension=".wav",
                                                  markDirPath="data/APLAWDW/valid_mark/",
                                                  markExtension=".marks")
dataPreprocessor.setDefaultRadius(defaultRadius = 40)
dataPreprocessor.process()
