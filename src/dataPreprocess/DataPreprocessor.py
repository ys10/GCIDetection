import h5py
import wave
import numpy
import os
import logging
from math import floor, ceil

''' Config the logger, output into log file.'''
log_file_name = "log/dataProcess.log"
if not os.path.exists(log_file_name):
    f = open(log_file_name, 'w')
    f.close()
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=log_file_name,
                    filemode='w')

''' Output to the console.'''
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


class DataPreprocessor(object):
    def __init__(self, dataFilePath, framSize=1, waveDirPath="data/wav/", waveExtension=".wav", markDirPath="data/mark/",
                 markExtension=".mark"):
        self.dataFilePath = dataFilePath
        self.frameSize = framSize
        self.waveDirPath = waveDirPath
        self.markDirPath = markDirPath
        self.waveExtension = waveExtension
        self.markExtension = markExtension
        pass

    def __getKeyList(self):
        # Get all mark file names as keys.
        items = os.listdir(self.markDirPath)
        keyList = []
        for names in items:
            if names.endswith(self.markExtension):
                keyList.append(names.split(".")[0])
                pass
            pass
        logging.debug("All files name: " + str(keyList))
        return keyList

    def __getDataType(self, sampleWidth):
        if(sampleWidth == 2):
            dataType = numpy.short
        else:
            dataType = numpy.int32
        return dataType

    def __getFramedLength(self, dataLength):
        length = int(ceil(dataLength/self.frameSize) * self.frameSize)
        return length

    def __padWave(self, waveData, dataType):
        length = self.__getFramedLength(waveData.__len__())
        logging.debug("length after padding:" + str(length))
        paddedData = numpy.zeros(shape = (length,), dtype=dataType)
        paddedData[:waveData.__len__()] = waveData
        return paddedData

    def __getLocations(self, markFile):
        locations = list()
        while 1:
            lines = markFile.readlines(100000)
            if not lines:
                break
            for line in lines:
                locations.append(float(line))
                pass
            pass
        return locations

    # Transform GCI locations to label(binary classification) sequence.
    def __transLocations2LabelSeq(self, locations, labelSeqLength, samplingRate):
        zero = numpy.zeros(shape=(labelSeqLength, 1), dtype=numpy.float32)
        one = numpy.ones(shape=(labelSeqLength, 1), dtype=numpy.float32)
        labelSeq = numpy.reshape(numpy.asarray([zero, one]).transpose(), [labelSeqLength, 2])
        logging.debug("mark data shape:" + str(labelSeq.shape))
        for location in locations:
            labelLocation = floor(int(location * samplingRate / self.frameSize)) - 1
            logging.debug("Time:" + str(labelLocation))
            labelSeq[labelLocation][0] = 1.0
            labelSeq[labelLocation][1] = 0.0
            pass
        return labelSeq

    def process(self):
        # Prepare an hdf5 file to save the process result
        with h5py.File(self.dataFilePath, 'w') as h5File:
            keyList = self.__getKeyList()
            # Iterate all files
            for fileName in keyList:
                logging.debug("File name:\t" + str(fileName))
                # Read wave params & data
                with wave.open(self.waveDirPath + fileName + self.waveExtension, "rb") as waveFile:
                    params = waveFile.getparams()
                    nChannels, sampleWidth, samplingRate, nSamplingPoints = params[:4]
                    logging.debug("\tnChannels:\t" + str(nChannels))
                    logging.debug("\tsampleWidth:\t" + str(sampleWidth))
                    logging.debug("\tsamplingRate:\t" + str(samplingRate))
                    logging.debug("\tnSamplingPoints:\t" + str(nSamplingPoints))
                    strWaveData = waveFile.readframes(nSamplingPoints)
                    dataType = self.__getDataType(sampleWidth)
                    waveData = numpy.fromstring(strWaveData, dtype=dataType)
                    framedLength = self.__getFramedLength(nSamplingPoints)
                    waveData = self.__padWave(waveData, dataType)
                    waveData = numpy.reshape(waveData, (framedLength, self.frameSize))
                    logging.debug("\twave data shape:\t" + str(waveData.shape))
                    # write wave date into hdf5 file.
                    h5File[fileName + "/input"] = list(waveData.astype(numpy.float32))
                    pass  # waveFile close
                # Read GCI locations.
                with open(self.markDirPath + fileName + self.markExtension) as markFile:
                    locations = self.__getLocations(markFile)
                    labelSeq = self.__transLocations2LabelSeq(locations, framedLength, samplingRate)
                    # write label date into hdf5 file.
                    h5File[fileName + "/label"] = labelSeq
                    pass  # markFile close
                pass  #
            pass  # h5File close
        pass

    pass


def main():
    dataFilePath = "data/hdf5/APLAWDW_test.hdf5"
    dataPreprocessor = DataPreprocessor(dataFilePath, framSize=1)
    dataPreprocessor.process()
    pass


if __name__ == '__main__':
    main()
    pass
