from src.dataPreprocess.DataPreprocessor import *

dataFilePath = "data/hdf5/APLAWDW_sampling_point_classification.hdf5"
dataPreprocessor = DataPreprocessor(dataFilePath,
                                    framSize=1,
                                    waveDirPath="data/APLAWDW/wav/",
                                    waveExtension=".wav",
                                    markDirPath="data/APLAWDW/mark/",
                                    markExtension=".marks")
dataPreprocessor.process()
