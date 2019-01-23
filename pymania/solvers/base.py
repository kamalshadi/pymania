from abc import ABC, abstractmethod
from pymania.primitives import *
import numpy as np
from .pipeline import *

class Solver(ABC):
    def __init__(self,name,backend,id,num_steps=5):
        self.name = name
        self._backend = backend
        self.id = id
        self.num_steps = num_steps
        self._subjects = {}
        self._rois = set([])
        self._loaded = False
        self.order = 0

    @property
    def subjects(self):
        return self._subjects

    @property
    def rois(self):
        return self._rois

    @property
    def backend(self):
        return self._backend

    def load(self):
        if (len(self.subjects)==0 or len(self.rois)==0):
            raise Exception('Please first add subjects and ROIs')
        for subject in self.subjects:
            data = self.backend.getdata_sts(subject,self.rois)
            self._subjects[subject]._sts = {(xx['n1'],xx['n2']):i for i,xx in enumerate(data)}
            self._subjects[subject]._data = [ST(subject,*xx.values()) for xx in data]
            self._subjects[subject]._rois = self.rois
        self._loaded = True

    def add_subject(self,id):
        self._subjects[id]=EnsembleST(subject=id)

    def add_roi(self,id):
        self._rois.add(id)

    def __iter__(self):
        for subject in self.subjects.values():
            yield subject

    def __len__(self):
        return len(self.subjects)

    def __call__(self,key):
        return self.subjects[key]

    def __getitem__(self,key):
        return self.subjects[key]

    @abstractmethod
    def find_noise_threshold(self):
        pass

    @abstractmethod
    def find_envelope_points(self):
        pass

    @abstractmethod
    def find_local_regressors(self):
        pass

    @abstractmethod
    def find_corrected_weights(self):
        pass

    @is_loaded
    def get_matrix1(self):
        '''
        Matrix1 elements are based on the strongest connected seed between two
        terminal ROIs
        '''
        for subject in self:
            l = len(subject.rois)
            mat = np.zeros((l,l))
            rois = sorted(subject.rois)
            for i,roi1 in enumerate(rois):
                for j,roi2 in enumerate(rois):
                    if roi1==roi2:continue
                    ind = subject._sts[(roi1,roi2)]
                    conn = subject.data[ind]
                    mat[i,j] = np.exp(conn.weight)*config.NOS
            subject.matrix1 = mat

    @is_loaded
    def run_mania1(self):
        '''
        Running MANIA on matrix1
        '''
        for subject in self:
            try:
                net,den,nar,t = mania_on_mat(subject.matrix1)
            except AttributeError:
                raise Exception("please first run get_matrix1 on solution")
            ind = np.argmin(nar)
            subject.threshold1 = t[ind]
            subject.mania1_network = net

    @is_loaded
    @pipeline(-1)
    def get_matrix2(self):
        '''
        Matrix2 elements are based on correction pipeline
        '''
        l = len(self.rois)
        rois = sorted(self.rois)
        for subject in self:
            mat = np.zeros((l,l))
            for i,roi1 in enumerate(rois):
                for j,roi2 in enumerate(rois):
                    if roi1==roi2:continue
                    ind = subject._sts[(roi1,roi2)]
                    conn = subject.data[ind]
                    tmp = min(conn.corrected_weight,0)
                    mat[i,j] = np.exp(tmp)*config.NOS
            subject.matrix2 = mat

    @is_loaded
    @pipeline(-1)
    def run_mania2(self,save=True):
        '''
        Running MANIA on matrix1
        '''
        for subject in self:
            try:
                net,den,nar,t = mania_on_mat(subject.matrix2)
            except AttributeError:
                raise Exception("please first run get_matrix2 on solution")
            ind = np.argmin(nar)
            subject.threshold2 = t[ind]
            subject.mania2_network = net
            if save:self.save_to_db(subject)


    @is_loaded
    @pipeline(-1)
    @abstractmethod
    def run(self, save=False):
        self.get_matrix1()
        self.get_matrix2()
        self.run_mania1()
        self.run_mania2()

    def save_to_db(self,subject):
        """Save the connections between all the ROIs to Neo4j database
        :return: None
        """
        rois = sorted(self.rois)
        for roi1 in tqdm(rois, desc='ROIs'):
            for roi2 in rois:
                if roi1 == roi2: continue
                ind = subject._sts[(roi1, roi2)]
                conn = subject.data[ind]
                attributes = {'SUBJECT':subject.subject,
                              'corrected_weight': conn.corrected_weight,
                              'corrected_weights': conn.corrected_weights,
                              'correction_type': conn.correction_type,
                              'noise_threshold': conn.noise_threshold,
                              'threshold1':subject.threshold1,
                              'threshold2':subject.threshold2,
                              'is_adjacent': conn.isAdjacent(True),
                              'is_connected': subject.is_connected(roi1, roi2),
                              'is_connected_mania1': subject.is_connected(roi1, roi2, mania2=False),
                              'regressor': conn.regressor.to_list(),
                              'envelope': conn.envelopes,
                              'weight': conn.weight,
                              'weights': conn.weights,
                              'regressor_type':conn.regressor_type
                              }

                attributes['run_id'] = self.name + '_' + self.id
                self.backend.write_connection(roi1, roi2, 'MANIA2', attributes)
