import h5py

class ResultEvaluator(object):
    def __init__(self, referenceFile, estimateFile, samplingRate):
        self.referenceFile = referenceFile
        self.estimateFile = estimateFile
        self.samplingRate = samplingRate
        self.defaultLarynxCycle = samplingRate / 300
        pass

    def evaluateResult(self, key):
        reference = self.referenceFile[key]
        estimate = self.estimateFile[key]
        
        #TODO
        pass

    def evaluateResults(self, keyList):
        # TODO
        pass

    pass