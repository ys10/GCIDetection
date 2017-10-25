import numpy as np
from dataAccessor.ResultWriter import *

class ClassificationResultWriter(ResultWriter):

    # Transform label(binary classification) sequence to GCI locations
    def transLabelSeq2Locations(self, labelSeq, samplingRate, frameSize, frameStride):
        locations = list()
        labelLocations = np.where(np.array(labelSeq)[:] == 1)[0].tolist()
        for labelLocation in labelLocations:
            # location = (labelLocation + 1) * frameSize / samplingRate
            location =(labelLocation * frameStride + frameSize / 2)/ samplingRate
            locations.append(location)
            pass
        return locations

    pass