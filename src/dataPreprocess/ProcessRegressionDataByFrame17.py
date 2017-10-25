dataFilePath = "data/hdf5/APLAWDW_frame_17_regression.hdf5"
dataPreprocessor = RegressionDataPreprocessor(dataFilePath,
                                              frameSize=17,
                                              frameStride=9,
                                              waveDirPath="data/APLAWDW/wav/",
                                              waveExtension=".wav",
                                              markDirPath="data/APLAWDW/mark/",
                                              markExtension=".marks")
dataPreprocessor.setGCICriticalDistance(gciCriticalDistance=40)
dataPreprocessor.process()
