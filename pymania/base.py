import numpy as np
import matplotlib.pyplot as plt
import pymania.utils as utils
import pymania.io as io



class MANIA2Error(Exception):
    pass

MIN_POINTS_ON_RIGHT = 5


class ST:
    g_noise_threshold = 5.8
    g_regressor = None
    min_r2 = 75
    min_envelope_points = 5
    min_points_on_right = 5
    def __init__(self,subject,roi1,roi2,_length = None,_weight = None):
        self.subject = subject
        self.roi1 = roi1
        self.roi2 = roi2
        if _length is not None and _weight is not None:
            self.data = zip(_length,_weight)
        else:
            tmp = io.getdata_st(subject,roi1,roi2)
            self.data = zip(tmp['_length'],tmp['_weight'])
        self._level = 0 # _level zero means no processing is yet done

    def __str__(self):
        if self.isNull():
            return f'S{self.subject}:{self.roi1}=>{self.roi2}:Null'
        return f'S{self.subject}:{self.roi1}=>{self.roi2}:{len(self)} seeds'

    def __repr__(self):
        return f'ST({self.subject},{self.roi1},{self.roi2})'


    @property
    def data(self):
        return self._data

    @data.setter
    def data(self,vec):
        self._data = np.array(sorted([[xx[0],np.log(xx[1]/5000.0)] for xx in vec if xx[1]>1]))


    @property
    def noise_threshold(self):
        if self._level>0:
            return self._noise_threshold
        raise AttributeError('Please first run find_noise_threshold')

    @property
    def envelopes(self):
        if self._level>1:
            return self._envelopes
        raise AttributeError('Please first run find_envelope_points')

    @property
    def regressor(self):
        if self._level>2:
            return self._regressor
        raise AttributeError('Please first run find_local_regressor')

    @regressor.setter
    def regressor(self,dic):
        self._regressor = utils.Regressor(dic['slope'],dic['intercept'],dic['r2'],dic['popt'])

    @property
    def local_corrected_weights(self):
        if self._level>3:
            return self._local_corrected_weights
        raise AttributeError('Please first run find_local_corrected_point')

    @property
    def local_corrected_weight(self):
        if self._level>3:
            return self._local_corrected_weight
        raise AttributeError('Please first run find_local_corrected_point')

    def __len__(self):
        return len(self.data)


    def find_noise_threshold(self):
        try:
            eta = utils.find_threshold(self.data)
            self._noise_threshold = eta
        except utils.NoiseError as e:
            print(e)
            self._noise_threshold = -np.log(1/5000)
        self._level = 1

    def noise_plot(self,ax = None):
        thresholds, eta = utils.sweep_threshold(self.data)
        threshold = thresholds[np.argmax(eta)]
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(15,10))
        else:
            assert len(ax)==2, "You must provide two axis for noise plot"
        self.plot(ax[0])
        ax[1].plot(thresholds,eta,'r*',ms=4,label=r"$\eta$")
        ax[1].axvline(threshold,ls='dashed',lw=2,label=r"$\eta^*$")
        ax[1].set_xlabel(r'$T_{log}$',fontsize=18)
        ax[1].set_ylabel(r'$\eta$',fontsize=18)
        ax[1].set_title('Noise analysis',fontsize=18)

        ax[1].set_xlim([-7.8,0])



    def find_envelope_points(self):
        outs = []
        l = len(self)
        for i,cur in enumerate(self.data):
            if i >= (l - MIN_POINTS_ON_RIGHT):
                break
            if self.data[i][1] < self.noise_threshold:
                continue
            right_side = self.data[(i+1):,:]
            tmp = right_side.max(axis=0)[1]
            if self.data[i][1] > tmp:
                outs.append(i)
        self._envelopes = outs
        self._level = 2
        return outs

    def isNull(self):
        return len(self)==0

    def max(self):
        tmp = np.argmax(self.data,axis=0)[1]
        return self.data[tmp,:]

    def find_local_regressor(self):
        if self.isNull():
            raise MANIA2Error('Connection is null')
        if self._level<2:
            raise MANIA2Error('Please first run find_envelope_points')
        if len(self.envelopes)<2:
            self.regressor = {slope:0,intercept:self.max()[1],r2:0,popt:(0,0)}
        self.regressor = utils.lslinear(self.data[self.envelopes,0],self.data[self.envelopes,1])
        self._level = 3

    def find_local_corrected_weight(self):
        if self._level<3:
            raise MANIA2Error('Please first run find_local_regressor')
        envs = self.data[self.envelopes,:]
        tmp = list(map(self.regressor.correct,envs))
        self._local_corrected_weights = tmp
        self._local_corrected_weight = np.median(tmp)
        self._level = 4


    def process(self):
        if self.isNull():
            raise MANIA2Error('Connection is null')
        self.find_noise_threshold()
        self.find_envelope_points()
        self.find_local_regressor()
        self.find_local_corrected_weight()


    def plot(self,ax = None):
        if self.isNull():
            print(f'Connection {self.roi1} to {self.roi2} is null')
            return
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,10))
        ax.plot(self.data[:,0],self.data[:,1],'b*')
        ax.set_xlabel('Distance (mm)',fontsize=18)
        ax.set_ylabel(r'$T_{log}$',fontsize=18)
        ax.set_xlabel('Distance (mm)',fontsize=18)
        ax.set_title(f'{self.roi1}->{self.roi2}',fontsize=18)
        if self._level>0:
            ax.axhline(self.noise_threshold,ls='--',color='r',lw=2,label="Noise threshold")
        if self._level>1:
            ax.plot(self.data[self.envelopes,0],self.data[self.envelopes,1],'gs',ms=6,label="Envelope points")
        if self._level>2:
            x = self.data[self.envelopes,0]
            z = list(map(self.regressor.predict,x))
            ax.plot(x,z,'k',lw=2,label='Local regressor')
            print(f'Local regressor R2 is {self.regressor.r2}')
        if self._level>3:
            l = len(self.local_corrected_weights)
            x = [0.0]*l
            ax.plot(x,self.local_corrected_weights,'ko',ms=4,label='Local corrected weights')
            ax.plot(0,self.local_corrected_weight,'k*',ms=6,label='Final corrected weight')
        ax.set_ylim(top=0)
        ax.set_xlim(left=-5)
        ax.axvline(0,lw=0.5,color='black')
        return ax

class PairST:
    def __init__(self,st1,st2):
        self.st1 = st1
        self.st2 = st2

    @property
    def st1(self):
        return self._st1

    @st1.setter
    def st1(self,arg):
        if isinstance(arg,ST):
            self._st1 = arg
        else:
            raise ValueError('Argument to PairST constructor is not an ST instance')

    @property
    def st2(self):
        return self._st2

    @st2.setter
    def st2(self,arg):
        if isinstance(arg,ST):
            self._st2 = arg
        else:
            raise ValueError('Argument to PairST constructor is not an ST instance')

    def plot(self,ax = None):
        if ax:
            assert len(ax)==2, "You must provide two axis for pairST plot"
        else:
            fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(15,10))
        self.st1.plot(ax[0])
        self.st2.plot(ax[1])

    def noise_plot(self, ax = None):
        if ax:
            m,n = ax.shape
            assert (m==2 and n==2), "You must provide 2x2 axis for pairST noise plot"
        else:
            fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(15,10))
        self.st1.noise_plot(ax[0,:])
        self.st2.noise_plot(ax[1,:])
        plt.tight_layout()




class EnsembleST:
    def __init__(self,arg,**kwargs):
        try:
            self.subject = kwargs['subject']
            self.mode = 0 # single subject mode
        except KeyError:
            self._mode = 1
        self.data = arg

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self,arg):
        tmp = arg[0]
        if isinstance(tmp,ST):
            self._data = arg
            self._rois = {(xx.roi1,xx.roi2):i for i,xx in enumerate(args)}
        elif isinstance(tmp,PairST):
            self._data = [tmp.st1]*2*len(arg)
            self._rois = {}
            for i,pair in enumerate(arg):
                self._data[2*i] = pair.st1
                self._data[2*i+1] = pair.st2
                self._rois[(pair.st1.roi1,pair.st1.roi2)] = 2*i
                self._rois[(pair.st2.roi1,pair.st2.roi2)] = 2*i + 1
        elif isinstance(tmp,str):
            assert self.mode==0, 'API only works in a single subject mode - Please specify subject ID -> (subject=ID)'
            tmp = io.getdata_sts(self.subject,arg)
            self._rois = {(xx['n1'],xx['n2']):i for i,xx in enumerate(tmp)}
            self._data = [ST(self.subject,*xx.values()) for xx in tmp]
        else:
            raise ValueError('Ensemble constructor arguments not known!')

    def __call__(self,roi1,roi2,pair=False):

        ind1 = self._rois[(roi1,roi2)]
        if not pair:
            return self.data[ind1]
        ind2 = self._rois[(roi2,roi1)]
        return PairST(self.data[ind1],self.data[ind2])


    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def find_roi_regressors(self):
        pass

    def plot_roi_regressor(self,roi):
        pass

    def find_corrected_weights(self):
        pass

    def get_matrix1(self):
        pass

    def get_matrix2(self):
        pass

    def run_mania1(self):
        pass

    def run_mania2(self):
        pass

    def describe(self):
        pass






# Get pair data for a subject
