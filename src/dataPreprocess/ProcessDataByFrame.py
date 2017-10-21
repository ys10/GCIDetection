from src.dataPreprocess.DataPreprocessor import *

dataFilePath = "data/hdf5/APLAWDW_frame_9.hdf5"
dataPreprocessor = DataPreprocessor(dataFilePath, framSize=9)
dataPreprocessor.process()
