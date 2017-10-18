import unittest
import h5py
from src.resultEvaluation.ResultEvaluator import *


class TestResultEvaluator(unittest.TestCase):
    def setUp(self):
        referenceFile = h5py.File("data/hdf5/APLAWDW.hdf5")
        estimateFile = h5py.File("data/hdf5/APLAWDW.hdf5")
        self.resultEvaluator = ResultEvaluator(referenceFile, estimateFile)
        pass

    def testEvaluateResults(self):
        # TODO
        pass

    def testCalCorrectRate(self):
        self.assertEqual(1, self.resultEvaluator.getGCICount())
        self.assertEqual(0, self.resultEvaluator.calCorrectRate())
        pass

    def testCalMissedRate(self):
        self.assertEqual(1, self.resultEvaluator.getGCICount())
        self.assertEqual(0, self.resultEvaluator.calMissedRate())
        pass

    def testCalFalseAlarmedRate(self):
        self.assertEqual(1, self.resultEvaluator.getGCICount())
        self.assertEqual(0, self.resultEvaluator.calFalseAlarmedRate())
        pass

    def testCalAcceptedRate(self):
        self.assertEqual(1, self.resultEvaluator.getGCICount())
        self.assertEqual(0, self.resultEvaluator.calAcceptedRate())
        pass

    def testCalIdentificationAccuracy(self):
        self.resultEvaluator.getErrorList().append([1, 1])
        self.assertEqual(0, self.resultEvaluator.calIdentificationAccuracy())
        pass

    pass


if __name__ == '__main__':
    unittest.main()
