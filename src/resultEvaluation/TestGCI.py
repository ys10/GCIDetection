import unittest
from src.resultEvaluation.GCI import *

class TestGCI(unittest.TestCase):

    def setUp(self):
        self.gci = GCI(625.1, 623.4, 626.6)
        pass

    def testAddEstimatedGCI(self):
        self.assertTrue(self.gci.addEstimatedGCI(624.2))
        self.assertFalse(self.gci.addEstimatedGCI(622.0))
        pass

    def testIsCorrect(self):
        self.gci.addEstimatedGCI(624.2)
        self.assertTrue(self.gci.isCorrect())
        pass

    def testIsMissed(self):
        self.gci.addEstimatedGCI(624.2)
        self.assertFalse(self.gci.isMissed())
        pass

    def testIsInLarynxCycle(self):
        self.assertTrue(self.gci.isInLarynxCycle(623.9))
        self.assertFalse(self.gci.isInLarynxCycle(623.1))
        pass

    def testCalError(self):
        self.gci.addEstimatedGCI(624.1)
        self.assertEqual(1, self.gci.calError())
        pass

    def testAcceptError(self):
        self.gci.addEstimatedGCI(624.2)
        self.assertFalse(self.gci.acceptError())
        pass

    def testTransRef2GCIList(self):
        reference = [625, 630]
        realGCIList = transRef2GCIList(reference)
        self.assertEqual(2, realGCIList.__len__())
        self.assertEqual(625, realGCIList[0].getLocation())
        self.assertEqual(620, realGCIList[0].getBorderLeft())
        self.assertEqual(627.5, realGCIList[0].getBorderRight())
        pass

    def testAssignEstimatedGCI(self):
        reference = [625, 630]
        realGCIList = transRef2GCIList(reference)
        estimate = [624, 626]
        assignEstimatedGCI(realGCIList, estimate)
        self.assertEqual(2, realGCIList[0].getEstimatedGCIList().__len__())
        self.assertEqual(0, realGCIList[1].getEstimatedGCIList().__len__())
        pass

    def testClassifyGCI(self):
        reference = [625, 630, 635, 640]
        realGCIList = transRef2GCIList(reference)
        estimate = [624, 626, 634.9, 641]
        assignEstimatedGCI(realGCIList, estimate)
        [correct, missed, falseAlarmed, accepted] = classifyGCI(realGCIList)
        self.assertEqual(2, correct.__len__())
        self.assertEqual(1, missed.__len__())
        self.assertEqual(1, falseAlarmed.__len__())
        self.assertEqual(1, accepted.__len__())
        pass

    def testGetErrorList(self):
        reference = [625, 630, 635, 640]
        realGCIList = transRef2GCIList(reference)
        estimate = [624, 626, 634.9, 641]
        assignEstimatedGCI(realGCIList, estimate)
        correct, _, _, _ = classifyGCI(realGCIList)
        errorList = getErrorList(correct)
        self.assertEqual(2, errorList.__len__())
        pass

    pass


if __name__ == '__main__':
    unittest.main()
