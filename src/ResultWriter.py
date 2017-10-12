class ResultWriter(object):
    def __init__(self, resultFile):
        self.resultFile = resultFile
        pass

    def saveBatchResult(self, batchResult, keyList):
        for i in range(keyList.__len__()):
            self.resultFile[keyList[i]] = batchResult[i]
            pass
        #TODO
        pass

    pass