# coding=utf-8

import h5py
import wave
import numpy
import os


def getMarkList(markFile):
    markList = list()
    while 1:
        lines = markFile.readlines(100000)
        if not lines:
            break
        for line in lines:
            markList.append(float(line))
            pass
        pass
    return markList


def instantSeq2SamplingPointSeq(instantList, samplingLength, samplingRate):
    lengthenTime = int(0.01 * samplingRate)
    zero = numpy.zeros(shape=(samplingLength + lengthenTime, 1), dtype=numpy.float32)
    one = numpy.ones(shape=(samplingLength + lengthenTime, 1), dtype=numpy.float32)
    markData = numpy.reshape(numpy.asarray([zero, one]).transpose(), [samplingLength + lengthenTime, 2])
    print("\tmark data shape:\t" + str(markData.shape))
    for instant in instantList:
        time = int(instant * samplingRate)
        markData[time][0] = 1.0
        markData[time][1] = 0.0
        pass
    return markData


waveDirPath = "data/wav/"
markDirPath = "data/mark/"
hdf5DirPath = "data/hdf5/"

waveExtension = ".wav"
markExtension = ".marks"
hdf5Extension = ".hdf5"

hdf5Filename = "APLAWDW"

# Get all file names
items = os.listdir(markDirPath)
filenameList = []
for names in items:
    if names.endswith(markExtension):
        filenameList.append(names.split(".")[0])
        pass
    pass
print("All files name: " + str(filenameList))

# Prepare an hdf5 file to save the process result
with h5py.File(hdf5DirPath + hdf5Filename + hdf5Extension, 'w') as h5File:
    # Iterate all files
    for fileName in filenameList:
        print("\nFile name:\t" + str(fileName))
        # Read wave params & data
        waveFile = wave.open(waveDirPath + fileName + waveExtension, "rb")
        params = waveFile.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        print("\tnchannels:\t" + str(nchannels))
        print("\tsampwidth:\t" + str(sampwidth))
        print("\tframerate:\t" + str(framerate))
        print("\tnframes:\t" + str(nframes))
        strData = waveFile.readframes(nframes)
        waveData = numpy.fromstring(strData, dtype=numpy.short)
        waveData = numpy.reshape(waveData, (nframes, 1))
        print("\twave data shape:\t" + str(waveData.shape))
        # waveData = (waveData + 32768.0) / 65536.0
        # write wave date into hdf5 file.
        h5File[fileName + "/input"] = list(waveData.astype(numpy.float32))
        waveFile.close()
        # read mark data
        with open(markDirPath + fileName + markExtension) as markFile:
            markList = getMarkList(markFile)
            # Initialize mark data with all zero list whose length equals to wave data
            markData = instantSeq2SamplingPointSeq(markList, nframes, framerate)
            # write mark date into hdf5 file.
            h5File[fileName + "/label"] = markData
        pass
