from dataPreprocess.RegressionDataPreprocessor import  *

dataFilePath = "data/hdf5/APLAWDW_frame_17_regression.hdf5"
dataPreprocessor = RegressionDataPreprocessor(dataFilePath,
                                    framSize=17,
                                    waveDirPath="data/APLAWDW/wav/",
                                    waveExtension=".wav",
                                    markDirPath="data/APLAWDW/mark/",
                                    markExtension=".marks")
dataPreprocessor.process()