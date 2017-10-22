import numpy as np
from dataAccessor.ResultWriter import *


class RegressionResultWriter(ResultWriter):
    # Transform label(binary classification) sequence to GCI locations
    def transLabelSeq2Locations(self, labelSeq, samplingRate, frameSize):
        locations = list()
        labelLocations = np.where(np.array(labelSeq)[:, 0] == 1)[0].tolist()
        for labelLocation in labelLocations:
            location = (labelLocation + 1) * frameSize / samplingRate
            locations.append(location)
            pass
        return locations

    pass