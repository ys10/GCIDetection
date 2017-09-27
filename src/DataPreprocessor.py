import h5py
import wave
import numpy
import os

waveDirPath = "data/wav/"
markDirPath = "data/mark/"
hdf5DirPath = "data/hdf5/"

waveExtension = ".wav"
markExtension = ".mark"
hdf5Extension = ".hdf5"

hdf5Filename = "APLAWDW_s_01_a"

# Get all file names
items = os.listdir(markDirPath)
filenameList = []
for names in items:
    if names.endswith(markExtension):
        filenameList.append(names.split(".")[0])
print(filenameList)

# Prepare an hdf5 file to save the process result
h5File = h5py.File(hdf5DirPath + hdf5Filename + hdf5Extension, 'w')

# Iterate all files
for fileName in filenameList:
    print("file name:" + str(fileName))
    # Read wave params & data
    waveFile = wave.open(waveDirPath + fileName + waveExtension, "rb")
    params = waveFile.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    print("nchannels: " + str(nchannels))
    print("sampwidth: " + str(sampwidth))
    print("framerate: " + str(framerate))
    print("nframes: " + str(nframes))
    strData = waveFile.readframes(nframes)
    waveData = numpy.fromstring(strData, dtype=numpy.short)
    waveData = numpy.reshape(waveData, (nframes, 1))
    print("wave data shape:"+str(waveData.shape))
    # waveData = (waveData + 32768.0) / 65536.0
    print(waveData)
    # write wave date into hdf5 file.
    h5File[fileName + "/input"] = list(waveData.astype(numpy.float32))
    waveFile.close()
    # read mark data
    markFile = open(markDirPath + fileName + markExtension)
    # Initialize mark data with all zero list whose length equals to wave data
    print("##################" + str(strData.__len__()))
    zero = numpy.zeros(shape = (strData.__len__(), 1), dtype=numpy.float32)
    # one = numpy.ones(shape = (strData.__len__(), 1), dtype=numpy.float32)
    # markData = numpy.reshape([zero, one], [strData.__len__(), 2])
    markData = numpy.reshape(zero, [strData.__len__(), 1])
    while 1:
        lines = markFile.readlines(100000)
        if not lines:
            break
        for line in lines:
            time = int(float(line) * framerate)
            # print(line)
            markData[time][0] = 1.0
            # markData[time][1] = 0.0
            pass
        pass
    # write mark date into hdf5 file.
    h5File[fileName + "/label"] = markData
    markFile.close()
    pass
h5File.close()