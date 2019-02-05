'''
This solver implements the Minimum Required Pipeline(MRP):
Envelope Points -> Local Regressor -> Correction
'''

from .common import *
from .base import Solver, RunId
from ..config import *
from .pipeline import *

class Constantine(Solver):
    '''Latest solver based on Constantine's meeting on January 18th'''
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
                    find_envelope_points(pair.st1,self.noise_threshold)
                    find_envelope_points(pair.st2,self.noise_threshold)

    def is_regressor_good(self, st, regressor):
        if self.run_id in [RunId.Sparse, RunId.VeryDense]:
            return regressor.r2 >= config.MIN_R2
        else:
            if regressor.r2 < config.MIN_R2:
                return False
            envs = st.data[st._envelopes, :]
            tmp = list(map(regressor.correct, envs))
            return np.median(tmp) <= 0

    @is_loaded
    @pipeline(3)
    def find_local_regressors(self):
        for subject in self:
            for roi1 in self.rois:
                for roi2 in self.rois:
                    if roi1==roi2:continue
                    pair = subject(roi1,roi2,True)
                    find_local_regressor(pair.st1)
                    find_local_regressor(pair.st2)
                    if self.is_regressor_good(pair.st1, pair.st1.regressor) and self.is_regressor_good(pair.st2, pair.st2.regressor):
                        pair.st1.regressor.is_good = True
                        pair.st2.regressor.is_good = True
                        pass
                    else:
                        bestR = create_null_regressor()

                        reg1 = pair.st1.regressor
                        reg1.kind = 'direction1'

                        reg2 = pair.st2.regressor
                        reg2.kind = 'direction2'

                        reg3 = find_local_regressor(pair,False)
                        reg3.kind = 'poolAll'

                        for reg in [reg1,reg2,reg3]:
                            if bestR.r2 < reg.r2:
                                if self.is_regressor_good(pair.st1, reg) and self.is_regressor_good(pair.st2, reg):
                                    bestR = reg
                                    bestR.is_good = True
                        pair.st1._regressor = bestR
                        pair.st2._regressor = bestR


    @is_loaded
    @pipeline(4)
    def find_corrected_weights(self):
        for subject in self:
            for st in subject:
                if self.run_id in [RunId.Sparse, RunId.VerySparse]:
                    find_corrected_weights(st, sparse=True)
                else:
                    find_corrected_weights(st, sparse=False)

    @is_loaded
    def run(self):
        self.find_noise_threshold()
        self.find_envelope_points()
        self.find_local_regressors()
        self.find_corrected_weights()
        super().run()
