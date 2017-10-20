import unittest

import h5py

from dataAccessor.ResultWriter import *


class TestResultWriter(unittest.TestCase):
    def setUp(self):
        self.resultFile = h5py.File("data/hdf5/TestResultWriter.hdf5", "w")
        self.resultWriter = ResultWriter(self.resultFile, samplingRate=2)
        pass

    def testSaveBatchResult(self):
        batchResult = [[[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [1, 0]]]
        keyList = ["testSaveBatchResult"]
        self.resultWriter.saveBatchResult(batchResult, keyList)
        pass

    def testTransLabelSeq2Locations(self):
        labelSeq = [[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1], [1, 0]]
        locations = self.resultWriter.transLabelSeq2Locations(labelSeq, 2)
        self.assertEqual(3, locations.__len__())
        self.assertEqual(1, locations[0])
        self.assertEqual(2, locations[1])
        self.assertEqual(3.5, locations[2])
        pass

    def tearDown(self):
        self.resultFile.close()
        pass

    pass


if __name__ == '__main__':
    unittest.main()
    pass
