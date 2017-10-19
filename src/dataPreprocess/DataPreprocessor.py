import h5py
import wave
import numpy
import os
import logging

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
    def __init__(self, dataFilePath, waveDirPath="data/wav/", waveExtension=".wav", markDirPath="data/mark/",
                 markExtension=".mark"):
        self.dataFilePath = dataFilePath
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
        lengthenTime = int(0.01 * samplingRate)
        zero = numpy.zeros(shape=(labelSeqLength + lengthenTime, 1), dtype=numpy.float32)
        one = numpy.ones(shape=(labelSeqLength + lengthenTime, 1), dtype=numpy.float32)
        labelSeq = numpy.reshape(numpy.asarray([zero, one]).transpose(), [labelSeqLength + lengthenTime, 2])
        logging.debug("mark data shape:" + str(labelSeq.shape))
        for location in locations:
            time = int(location * samplingRate)
            labelSeq[time][0] = 1.0
            labelSeq[time][1] = 0.0
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
                    nchannels, sampwidth, framerate, nframes = params[:4]
                    logging.debug("\tnchannels:\t" + str(nchannels))
                    logging.debug("\tsampwidth:\t" + str(sampwidth))
                    logging.debug("\tframerate:\t" + str(framerate))
                    logging.debug("\tnframes:\t" + str(nframes))
                    strData = waveFile.readframes(nframes)
                    waveData = numpy.fromstring(strData, dtype=numpy.short)
                    waveData = numpy.reshape(waveData, (nframes, 1))
                    logging.debug("\twave data shape:\t" + str(waveData.shape))
                    # write wave date into hdf5 file.
                    h5File[fileName + "/input"] = list(waveData.astype(numpy.float32))
                    pass  # waveFile close
                # Read GCI locations.
                with open(self.markDirPath + fileName + self.markExtension) as markFile:
                    locations = self.__getLocations(markFile)
                    # Initialize mark data with all zero list whose length equals to wave data
                    labelSeq = self.__transLocations2LabelSeq(locations, nframes, framerate)
                    # write mark date into hdf5 file.
                    h5File[fileName + "/label"] = labelSeq
                    pass  # markFile close
                pass  #
            pass  # h5File close
        pass

    pass


def main():
    dataFilePath = "data/hdf5/APLAWDW_test.hdf5"
    dataPreprocessor = DataPreprocessor(dataFilePath)
    dataPreprocessor.process()
    pass

if __name__ == '__main__':
    main()
    pass