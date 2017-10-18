import numpy as np

from resultEvaluation.GCI import *


class ResultEvaluator(object):
    def __init__(self, referenceFile, estimateFile, defaultRadius=5):
        self.referenceFile = referenceFile
        self.estimateFile = estimateFile
        self.defaultRadius = defaultRadius
        self.gciCount = 0
        self.correctCount = 0
        self.missedCount = 0
        self.falseAlarmedCount = 0
        self.acceptedCount = 0
        self.errorList = list()
        pass

    def getGCICount(self):
        return self.gciCount

    def getCorrectCount(self):
        return self.correctCount

    def getMissedCount(self):
        return self.missedCount

    def getFalseAlarmedCount(self):
        return self.falseAlarmedCount

    def getAcceptedCount(self):
        return self.acceptedCount

    def getErrorList(self):
        return self.errorList

    def __evaluateResult(self, key):
        #  List the location of  real GCIs.
        reference = self.referenceFile[key]
        #  List the location of  estimated GCIs.
        estimate = self.estimateFile[key]
        # Transform a list of reference gci location to a list of  GCI class.
        realGCIList = transRef2GCIList(reference)
        self.gciCount += realGCIList.__len__()
        # Assign estimated GCIs to real GCIs.
        assignEstimatedGCI(realGCIList, estimate)
        # Classify assigned  GCIs into three classes: correct, missed, falseAlarmed.
        [correct, missed, falseAlarmed, accepted] = classifyGCI(realGCIList)
        self.correctCount += correct.__len__()
        self.missedCount += missed.__len__()
        self.falseAlarmedCount += falseAlarmed.__len__()
        self.acceptedCount += accepted.__len__()
        # Get a list of error  between real GCI & estimated GCI
        errorList = getErrorList(correct)
        self.errorList.append(errorList)
        pass

    def evaluateResults(self, keyList):
        for key in keyList:
            self.__evaluateResult(key)
            pass
        pass

    def calCorrectRate(self):
        return self.correctCount / self.gciCount

    def calMissedRate(self):
        return self.missedCount / self.gciCount

    def calFalseAlarmedRate(self):
        return self.falseAlarmedCount / self.gciCount

    def calAcceptedRate(self):
        return self.acceptedCount / self.correctCount

    def calIdentificationAccuracy(self):
        return np.std(self.errorList, ddof=1)

    pass
