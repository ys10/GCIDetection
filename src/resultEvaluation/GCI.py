import numpy as np


class GCI(object):
    def __init__(self, location, borderLeft, borderRight):
        self.location = location
        self.borderLeft = borderLeft
        self.borderRight = borderRight
        self.estimatedGCIList = list()
        pass

    def getLocation(self):
        return self.location

    def getBorderLeft(self):
        return self.borderLeft

    def getBorderRight(self):
        return self.borderRight

    def getEstimatedGCIList(self):
        return self.estimatedGCIList

    def addEstimatedGCI(self, location):
        if self.isInLarynxCycle(location):
            self.estimatedGCIList.append(location)
            return True
        return False

    def isCorrect(self):
        if self.estimatedGCIList.__len__() == 1:
            return True
        return False

    def isMissed(self):
        if self.estimatedGCIList.__len__() == 0:
            return True
        return False

    def isInLarynxCycle(self, location):
        if location >= self.borderLeft and location < self.borderRight:
            return True
        return False

    def calError(self):
        if self.isCorrect():
            return abs(self.location - self.estimatedGCIList[0])
        return 0

    def acceptError(self, admissibleError=0.25):
        if self.isCorrect():
            if admissibleError >= self.calError():
                return True
        return False

    pass


# Transform a list of reference gci location to a list of  GCI class.
def transRef2GCIList(reference, defaultRadius=5):
    realGCIList = list()
    for i in range(0, reference.__len__()):
        leftRadius = defaultRadius \
            if i == 0 or defaultRadius <= (reference[i] - reference[i - 1]) / 2 \
            else (reference[i] - reference[i - 1]) / 2
        rightRadius = defaultRadius \
            if i == reference.__len__() - 1 or defaultRadius <= (reference[i + 1] - reference[i]) / 2 \
            else (reference[i + 1] - reference[i]) / 2
        realGCIList.append(GCI(reference[i], reference[i] - leftRadius, reference[i] + rightRadius))
        pass
    return realGCIList


# Assign estimated GCIs to real GCIs.
def assignEstimatedGCI(realGCIList, estimate):
    lastGCIIndex = 0
    for i in range(0, estimate.__len__()):
        for j in range(lastGCIIndex, realGCIList.__len__()):
            if realGCIList[j].isInLarynxCycle(estimate[i]):
                realGCIList[j].addEstimatedGCI(estimate[i])
                lastGCIIndex = j
                pass
            pass
        pass
    pass


# Classify assigned  GCIs into three classes: correct, missed, falseAlarmed.
def classifyGCI(realGCIList):
    correct = list()
    missed = list()
    falseAlarmed = list()
    accepted = list()
    for gci in realGCIList:
        if gci.isCorrect():
            correct.append(gci)
            if gci.acceptError():
                accepted.append(gci)
        elif gci.isMissed():
            missed.append(gci)
        else:
            falseAlarmed.append(gci)
        pass
    return [correct, missed, falseAlarmed, accepted]


# Get a list of error  between real GCI & estimated GCI.
def getErrorList(correctList):
    errorList = list()
    for gci in correctList:
        errorList.append(gci.calError())
        pass
    return errorList


def transGCIList2GCIMarkMatrix(gciList, timeSteps):
    maskMatrix = np.zeros(shape=(gciList.__len__(), timeSteps), dtype=np.short)
    for i in range(gciList.__len__()):
        maskStart = int(gciList[i].getBorderLeft())
        maskEnd = int(gciList[i].getBorderRight())
        maskMatrix[i][maskStart:maskEnd] = 1
        pass
    return list(maskMatrix)
