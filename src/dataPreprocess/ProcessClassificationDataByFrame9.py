from dataPreprocess.ClassificationDataPreprocessor import  *

dataFilePath = "data/hdf5/APLAWDW_frame_9_classification.hdf5"
dataPreprocessor = ClassificationDataPreprocessor(dataFilePath,
                                    framSize=9,
                                    waveDirPath="data/APLAWDW/wav/",
                                    waveExtension=".wav",
                                    markDirPath="data/APLAWDW/mark/",
                                    markExtension=".marks")
dataPreprocessor.process()
