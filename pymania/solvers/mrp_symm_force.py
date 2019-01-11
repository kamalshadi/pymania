'''
This solver implements the Minimum Required Pipeline(MRP):
Envelope Points -> Local Regressor -> Correction
'''

from .common import *
from .base import Solver
from ..config import *
from .pipeline import *

class MRP_F(Solver):
    '''Minimum require pipeline with no symmetry force'''
    def __init__(self,backend,id):
        super().__init__(self.__class__.__name__,backend,id)

    @is_loaded
    @pipeline(1)
    def find_noise_threshold(self):
        self.noise_threshold = find_noise_threshold()
        for subject in self:
            subject._noise_threshold = self.noise_threshold
            for st in subject:
                st._noise_threshold = self.noise_threshold

    @is_loaded
    @pipeline(2)
    def find_envelope_points(self):
        for subject in self:
            for roi1 in self.rois:
                for roi2 in self.rois:
                    if roi1==roi2:continue
                    pair = subject(roi1,roi2,True)
                    find_envelope_points(pair,self.noise_threshold)

    @is_loaded
    @pipeline(3)
    def find_local_regressors(self):
        for subject in self:
            for roi1 in self.rois:
                for roi2 in self.rois:
                    if roi1==roi2:continue
                    pair = subject(roi1,roi2,True)
                    find_local_regressor(pair)

    @is_loaded
    @pipeline(4)
    def find_corrected_weights(self):
        for subject in self:
            for st in subject:
                find_corrected_weights(st)

    @is_loaded
    def run(self):
        self.find_noise_threshold()
        self.find_envelope_points()
        self.find_local_regressors()
        self.find_corrected_weights()
        super().run()
