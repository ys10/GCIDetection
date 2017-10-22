from dataPreprocess.ClassificationDataPreprocessor import  *

dataFilePath = "data/hdf5/APLAWDW_frame_17_classification.hdf5"
dataPreprocessor = ClassificationDataPreprocessor(dataFilePath,
                                    framSize=17,
                                    waveDirPath="data/APLAWDW/wav/",
                                    waveExtension=".wav",
                                    markDirPath="data/APLAWDW/mark/",
                                    markExtension=".marks")
dataPreprocessor.process()
