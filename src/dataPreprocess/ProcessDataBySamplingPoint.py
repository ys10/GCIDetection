from src.dataPreprocess.DataPreprocessor import *

dataFilePath = "data/hdf5/APLAWDW_frame.hdf5"
dataPreprocessor = DataPreprocessor(dataFilePath, framSize=1)
dataPreprocessor.process()
