from abc import abstractmethod

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
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=log_file_name,
                    filemode='w')

''' Output to the console.'''
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


class DataPreprocessor(object):
    def __init__(self, dataFilePath, frameSize=1, frameStride=1, waveDirPath="data/wav/", waveExtension=".wav",
                 markDirPath="data/mark/",
                 markExtension=".marks"):
        self.dataFilePath = dataFilePath
        self.frameSize = frameSize
        self.frameStride = frameStride
        self.waveDirPath = waveDirPath
        self.markDirPath = markDirPath
        self.waveExtension = waveExtension
        self.markExtension = markExtension
        pass

    def getKeyList(self):
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

    def getLocations(self, markFile):
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

    def getDataType(self, sampleWidth):
        if (sampleWidth == 2):
            dataType = numpy.short
        else:
            dataType = numpy.int32
        return dataType

    def getFramedLength(self, dataLength):
        count = self.getFrameCount(dataLength)
        length = (count - 1) * self.frameStride + self.frameSize
        return length

    def getFrameCount(self, dataLength):
        count = int(ceil((dataLength - self.frameSize) / self.frameStride)) + 1
        return count

    def getLabelIndex(self, location, samplingRate, frameCount):
        if (location * samplingRate  >= self.frameStride * frameCount):
            labelIndex = frameCount-1
            pass
        else:
            labelIndex = floor(int(location * samplingRate / self.frameStride))
        return labelIndex

    def padWave(self, waveData, dataType):
        length = self.getFramedLength(waveData.__len__())
        logging.debug("length after padding:" + str(length))
        paddedData = numpy.zeros(shape=(length,), dtype=dataType)
        paddedData[:waveData.__len__()] = waveData
        return paddedData

    def getFramedInput(self, waveData, samplingRate):
        waveData = list(waveData.astype(numpy.float32))
        framedCount = self.getFrameCount(waveData.__len__())
        input = list()
        for i in range(framedCount):
            startIndex = i * self.frameStride
            frameData = waveData[startIndex: startIndex + self.frameSize]
            input.append(frameData)
            pass
        return input

    def process(self):
        # Prepare an hdf5 file to save the process result
        with h5py.File(self.dataFilePath, 'w') as h5File:
            keyList = self.getKeyList()
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
                    dataType = self.getDataType(sampleWidth)
                    waveData = numpy.fromstring(strWaveData, dtype=dataType)
                    framedLength = self.getFramedLength(nSamplingPoints)
                    waveData = self.padWave(waveData, dataType)
                    # waveData = numpy.reshape(waveData, (framedLength, self.frameSize))
                    # logging.debug("\twave data shape:\t" + str(waveData.shape))
                    input = self.getFramedInput(waveData, samplingRate)
                    # write wave date into hdf5 file.
                    h5File[fileName + "/input"] = input
                    pass  # waveFile close
                # Read GCI locations.
                with open(self.markDirPath + fileName + self.markExtension) as markFile:
                    locations = self.getLocations(markFile)
                    logging.debug("gci locations:" + str(locations))
                    '''Process locations to label sequence.'''
                    frameCount = self.getFrameCount(framedLength)
                    labelSeq = self.transLocations2LabelSeq(locations, frameCount, samplingRate)
                    # write label date into hdf5 file.
                    h5File[fileName + "/label"] = labelSeq
                    '''Process locations to gci mask.'''
                    # mask = self.transLocations2GCIMask(locations, framedLength, samplingRate)
                    # if mask is not None:
                    #     h5File[fileName + "/mask"] = mask
                    #     pass
                    '''Process gci count.'''
                    gciCount = [len(locations)]
                    logging.debug("gciCount:" + str(gciCount))
                    h5File[fileName + "/gciCount"] = numpy.asarray(gciCount, dtype=numpy.float32)
                    pass  # markFile close
                pass  #
            pass  # h5File close
        pass

    # Transform GCI locations to label(binary classification) sequence.
    @abstractmethod
    def transLocations2LabelSeq(self, locations, labelSeqLength, samplingRate):
        pass

    @abstractmethod
    def transLocations2GCIMask(self, locations, frameCount, samplingRate):
        pass

    pass
