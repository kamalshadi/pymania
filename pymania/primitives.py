import matplotlib.pyplot as plt
from .utils import *
from .io import *
from .config import *
from tqdm import tqdm
import pickle as pk


class MANIA2Error(Exception):
    pass


MIN_POINTS_ON_RIGHT = 5

verbose = False


class ST:
    """
    ST class contains all data points from probtrackx run, namely, length and
    fraction of streamlines reaching from roi1(source) to roi2(target)
    """
    roi_regressors = {}  # dictionary if roi_regressors keyed by subjects
    min_r2 = 75
    min_envelope_points = 5
    min_points_on_right = 5

    @staticmethod
    def load_roi_regressors():
        D = {}
        for roi in ['L'+str(i) for i in range(1,181)]:
            tmp = get_roi_regressor(roi)
            for sub in tmp.keys():
                try:
                    D[int(sub)]['s-'+roi] = Regressor(*tmp[sub]['s'])
                    D[int(sub)]['t-'+roi] = Regressor(*tmp[sub]['t'])
                except KeyError:
                    D[int(sub)] = {}
                    D[int(sub)]['s-'+roi] = Regressor(*tmp[sub]['s'])
                    D[int(sub)]['t-'+roi] = Regressor(*tmp[sub]['t'])
        ST.roi_regressors = D

    def __init__(self, subject, roi1, roi2, _length=None, _weight=None, border=0):
        self.subject = subject
        self.roi1 = roi1
        self.roi2 = roi2
        if _length is not None and _weight is not None:
            self.data = zip(_length, _weight)
        else:
            tmp = getdata_st(subject, roi1, roi2)
            self.data = zip(tmp['_length'], tmp['_weight'])
        self.border = border
        self._level = 0  # _level zero means no processing is yet done
        self._mania_loaded = False


    def load_mania_results(self):
        tmp =getmania_st(self.subject,self.roi1,self.roi2)
        self.correction_type = tmp['correction_type']
        self._is_connected = tmp['is_connected']
        self._is_connected_mania1 = tmp['is_connected_mania1']
        self._threshold1 = tmp['threshold1']
        self._threshold2 = tmp['threshold2']
        self._corrected_weights = tmp['corrected_weights']
        self._mania_loaded = True


    def __str__(self):
        if self.isNull():
            return f'S{self.subject}:{self.roi1}=>{self.roi2}:Null'
        return f'S{self.subject}:{self.roi1}=>{self.roi2}:{len(self)} seeds'

    def __repr__(self):
        return f'ST({self.subject},{self.roi1},{self.roi2})'

    def __iter__(self):
        for data in self.data:
            yield data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, vec):
        self._data = np.array(sorted([[xx[0], np.log(xx[1]/5000.0)] for xx in vec if xx[1] > 1]))

    @property
    def threshold1(self):
        if self._mania_loaded:
            return self._threshold1
        raise MANIA2Error('Please run load_mania_results first')

    @property
    def threshold2(self):
        if self._mania_loaded:
            return self._threshold2
        raise MANIA2Error('Please run load_mania_results first')


    @property
    def weights(self):
        if self.isNull():
            return []
        return self.data[:, 1]

    @property
    def weights(self):
        if self.isNull():
            return []
        return self.data[:, 1]

    @property
    def weight(self):
        if self.isNull():
            return np.log(1/config.NOS)
        return np.max(self.data[:, 1])

    @property
    def noise_threshold(self):
        return self._noise_threshold

    @property
    def envelopes(self):
        if self.isNull():
            return []
        try:
            return self._envelopes
        except AttributeError:
            raise AttributeError('Please first run find_envelope_points of the solver')

    @property
    def regressor(self):
        try:
            return self._regressor
        except AttributeError:
            raise AttributeError('Please first run find_local_regressors of the solver')

    @regressor.setter
    def regressor(self,dic):
        self._regressor = Regressor(dic['slope'],dic['intercept'],dic['r2'],dic['ME'])

    @property
    def corrected_weights(self):
        try:
            return self._corrected_weights
        except AttributeError:
            raise MANIA2Error(f'Weights are not yet corrected for {self.roi1} to {self.roi2}')

    @corrected_weights.setter
    def corrected_weights(self,arg):
        self._corrected_weights = arg

    @property
    def corrected_weight(self):
        try:
            return self._corrected_weight
        except AttributeError:
            raise MANIA2Error("Corrected weight not found")

    def isConnected(self,mania2=True):
        if mania2:
            if self.corrected_weight is not None:
                return self.corrected_weight >= self.threshold2
            else:
                return None
        else:
            if self.weight is not None:
                return self.weight >= self.threshold1
            else:
                return None


    def find_noise_threshold(self):
        try:
            eta = find_threshold(self.data)
            self._noise_threshold = eta
        except NoiseError as e:
            if verbose:
                print(e)
            self._noise_threshold = -np.log(1/5000)
        self._level = 1

    def noise_plot(self,ax = None):
        thresholds, eta = sweep_threshold(self.data)
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


    def isNull(self):
        '''
        ST with no nozero data point is considered Null
        '''
        return len(self)==0

    def isAdjacent(self,strong=False):
        '''
        ST a->b is adjacent if a and b have border with each other.
        They are strongly adjacent if the border seeds are the strongest
        connection between them.
        '''
        if not strong:
            return self.border is not None and self.border>0
        if self.border<1:
            return False
        for i,w in enumerate(self.data):
            if w[0]>1:return False # Seeds are not adjacent anymore
            if i==(len(self)-1): return False # Last seed
            if w[1]> (np.max(self.data[(i+1):,:],axis=0)[1]):
                #adjacent seed with maximum connection weight
                return True
        return False


    def max(self):
        '''
        The seed with maxim weight to from the source to target.
        '''
        tmp = np.argmax(self.data,axis=0)[1]
        return self.data[tmp,:]

    def argmax(self):
        '''
        The index of the seed with maximum weight from the source to the target.
        '''
        return np.argmax(self.data,axis=0)[1]

    def axonal_distance(self,threshold=5):
        t = np.log(threshold/config.NOS)
        if self.isAdjacent(False):return 0
        for l,w in self.data:
            if w>t:
                return l
        return np.nan


    def process(self):
        if self.isNull():
            raise MANIA2Error('Connection is null')
        self.find_noise_threshold()
        self.find_envelope_points()
        self.find_local_regressor()
        self.find_local_corrected_weight()

    def get_type(self):
        sparse = False
        if self.correction_type == 'Null':
            return 'Null'
        if self.correction_type == 'Adjacent':
            return 'Adjacent'
        if self.correction_type in ['Bad regressor', 'No Envelope No regress']:
            return 'No Correction'
        if self.correction_type == 'Regress' and self.regressor.kind == 'independent':
            return 'Independent'
        if sparse:
            if self.regressor.kind in ['direction1', 'direction2', 'poolAll', 'poolEnvelopes']:
                return 'No Correction'
        else:
            if self.regressor.kind == 'direction1' or self.regressor.kind == 'direction2':
                return 'one direction'
            if self.regressor.kind == 'poolAll' or self.regressor.kind == 'poolEnvelopes':
                return self.regressor.kind
        if self.correction_type == 'No Correction':
            return 'No Correction'
        print('Error while evaluating: ', self.correction_type, self.regressor.kind)

    def plot(self,ax = None):
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,10))

        ax.set_xlabel('Distance (mm)',fontsize=18)
        ax.set_ylabel(r'$T_{log}$',fontsize=18)
        ax.set_xlabel('Distance (mm)',fontsize=18)
        ax.set_title(f'{self.roi1}->{self.roi2}-({self.get_type()})',fontsize=18)
        if self.isNull():
            ax.set_facecolor('#000000')
            ax.text(0.05,.9,f'SUB:{self.subject}',color="white")
            return
        ax.text(0.05,-1,f'SUB:{self.subject}',color="black")
        ax.plot(self.data[:,0],self.data[:,1],'b*')
        if(len(self.envelopes)>0):
            ax.plot(self.data[self.envelopes,0],self.data[self.envelopes,1],'gs',ms=6,label="Envelope points")
            if self.regressor:
                x = self.data[self.envelopes,0]
                z = list(map(self.regressor.predict,x))
                ax.plot(x,z,'k',lw=2,label='Local regressor')
            # if (self.correction_type == 'envelope' or self.correction_type == 'above noise'):
            #     try:
            #         R = ST.roi_regressors[self.subject][f's-{self.roi1}']
            #         if len(x)>0:
            #             c = R.correct([x[-1],z[-1]])
            #         else:
            #             t = self.max()
            #             x = [t[0]]
            #             z = [t[1]]
            #             c = R.correct([x[-1],z[-1]])
            #         ax.plot([0,x[-1]],[c,z[-1]],'r--',lw=2,label='SF regressor')
            #     except KeyError:
            #         pass
            #     try:
            #         R = ST.roi_regressors[self.subject][f't-{self.roi2}']
            #         if len(x)>0:
            #             c = R.correct([x[-1],z[-1]])
            #         else:
            #             t = self.max()
            #             x = [t[0]]
            #             z = [t[1]]
            #             c = R.correct([x[-1],z[-1]])
            #         ax.plot([0,x[-1]],[c,z[-1]],'m--',lw=2,label='TF regressor')
            #     except KeyError:
            #         pass

        if self.isConnected():
            ax.set_facecolor((0/255,255/255,0/255,.2))
        else:
            ax.set_facecolor((80/255,80/255,80/255,.2))

        ax.set_ylim(top=0)
        ax.set_xlim(left=-5)
        ax.axvline(0,lw=0.5,color='black')
        # ax.axhline(np.log(self.threshold2/NOS),lw=2,color='magenta',label='MANIA2 threshold')
        # ax.axhline(np.log(self.threshold1/NOS),lw=2,ls='dashed',color='magenta',label='MANIA1 threshold')
        return ax

class PairST:
    '''
    PairST class contains an ST instance and its reverse, i.e., a->b and b->a STs
    '''
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
        if ax is not None:
            assert len(ax)==2, "You must provide two axis for pairST plot"
        else:
            fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(15,10))
        self.st1.plot(ax[0])
        self.st2.plot(ax[1])

    def noise_plot(self, ax = None):
        if ax is not None:
            m,n = ax.shape
            assert (m==2 and n==2), "You must provide 2x2 axis for pairST noise plot"
        else:
            fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(15,10))
        self.st1.noise_plot(ax[0,:])
        self.st2.noise_plot(ax[1,:])
        plt.tight_layout()




class EnsembleST:
    '''
    EnsembleST is mainly designed to contain all STs for a single subject study.
    In future, we may extend this class to contain a single ST
    from across a subject cohort.
    '''
    def __init__(self,arg=None,ih=False,**kwargs):
        try:
            self.subject = kwargs['subject']
            self.mode = 0 # single subject mode
        except KeyError:
            self._mode = 1
        if arg:
            self.data = arg
        self.ih = ih
        self.roi_regressors = None

    @property
    def rois(self):
        return self._rois

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self,arg):
        tmp = arg[0]
        if isinstance(tmp,ST):
            self._data = arg
            self._sts = {(xx.roi1,xx.roi2):i for i,xx in enumerate(arg)}
            self._rois = set([item for sublist in self._sts.keys() for item in sublist])
        else:
            raise ValueError('Ensemble constructor arguments not known!')


    def run_ranks(self,corrected=False,sparse=False):
        def aux(weight):
            tmp = int(config.NOS*np.exp(weight))
            return min(tmp,config.NOS)

        if not corrected:
            ranked = sorted([(xx.roi1,xx.roi2,aux(xx.weight)) for xx in self.data],key=lambda x:x[-1],reverse=True)
        else:
            if sparse:
                ranked = sorted(
                [(xx.roi1,xx.roi2,aux(xx.corrected_weight))\
                if xx.regressor.kind=='independent' else (xx.roi1,xx.roi2,aux(xx.weight))\
                for xx in self.data],
                key=lambda x:x[-1],reverse=True)
            else:
                ranked = sorted(
                [(xx.roi1,xx.roi2,
                aux(xx.corrected_weight)) for xx in self.data],
                key=lambda x:x[-1],reverse=True)
        self.ranked_sts = ranked
        self.ranks = {}
        for i,st in enumerate(ranked):
            self.ranks[(st[0],st[1])] = (i,st[2])

    def proportional_net(self,density_threshold=0.10):
        self.run_ranks()
        ind = int(len(self)*density_threshold)
        N = len(self.rois)
        inds = {xx:i for i,xx in enumerate(sorted(self.rois))}
        net = np.zeros((N,N))
        for i,rois in enumerate(self.ranked_sts[0:ind]):
            r1 = rois[0]
            r2 = rois[1]
            ind1 = inds[r1]
            ind2 = inds[r2]
            net[ind1,ind2] = 1
        return net




    def __call__(self,roi1,roi2,pair=False):
        '''
        Get the specific ST or PairST from the ensemble
        '''
        ind1 = self._sts[(roi1,roi2)]
        if not pair:
            return self.data[ind1]
        ind2 = self._sts[(roi2,roi1)]
        return PairST(self.data[ind1],self.data[ind2])


    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for st in self.data:
            yield st

    def noise_spectrum(self):
        for st in self.data:
            if st._level<1:
                st.find_noise_threshold()
        out = [xx.noise_threshold for xx in self.data]
        return out

    def plot_noise_spectrum(self,ax = None):
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15,10))
        values = self.noise_spectrum()
        f,e = np.histogram(values, bins=50)
        ax.plot(e[1:],f,'b-',lw=2)
        ax.set_xlabel('Threshold',fontsize=18)
        ax.set_ylabel('Frequency',fontsize=18)
        ax.set_title('Noise thresholds',fontsize=18)
        return ax


    def regressor_spectrum(self):
        for st in self.data:
            if st._level<1:
                st.find_noise_threshold()
            if st._level<2:
                st.find_local_regressor()
        out = [xx.regressor for xx in self.data]
        return out


    def plot_regressor_spectrum(self,ax = None):
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(15,10))
        values = self.regressor_spectrum()
        f,e = np.histogram([xx.slope for xx in values], bins=50)
        ax[0].plot(e[1:],f,'b-',lw=2)
        ax[0].set_xlabel('Slope',fontsize=18)
        ax[0].set_ylabel('Frequency',fontsize=18)
        ax[0].set_title('Slope distribution',fontsize=18)
        f,e = np.histogram([xx.r2 for xx in values], bins=50)
        ax[1].plot(e[1:],f,'b-',lw=2)
        ax[1].set_xlabel(r'$r^2$',fontsize=18)
        ax[1].set_ylabel('Frequency',fontsize=18)
        ax[1].set_title(r'$r^2$ distribution',fontsize=18)
        return ax

    def preprocess(self):
        '''
        Local processing for each a->b ST relation in the ensemble
        '''
        print(f'Preprocessing connections for subject{self.subject}')
        for st in tqdm(self.data,total=len(self),desc='Pre'):
            if st.isNull():continue
            st.process()


    def find_roi_regressors(self):
        print(f'Find roi regressors for subject{self.subject}')
        _Rs = {}
        _Rt = {}
        for roi in tqdm(self.rois,total=len(self.rois)):
            # source fixed regressor
            xs = []
            ys = []
            for i,xx in enumerate(self.rois):
                if xx==roi:continue
                ind = self._sts[(roi,xx)]
                can = self.data[ind]
                if (can!=roi and can.regressor.r2>75 and len(can.envelopes)>5):
                    x = [can.data[xx][0] for xx in can.envelopes]
                    xm = np.mean(x)
                    y = [can.data[xx][1] for xx in can.envelopes]
                    ym = np.mean(y)
                    xs = xs + [xx-xm for xx in x]
                    ys = ys + [xx-ym for xx in y]
            try:
                F = lslinear(xs,ys)
            except RegressionError:
                F = {'slope':0,'intercept':-np.log(1/NOS),'r2':0}
            _Rs["s-"+roi] = Regressor(F['slope'],F['intercept'],F['r2'])
            # target fixed regressor
            xs = []
            ys = []
            for i,xx in enumerate(self.rois):
                if xx==roi:continue
                ind = self._sts[(xx,roi)]
                can = self.data[ind]
                if (can!=roi and can.regressor.r2>75 and len(can.envelopes)>5):
                    x = [can.data[xx][0] for xx in can.envelopes]
                    xm = np.mean(x)
                    y = [can.data[xx][1] for xx in can.envelopes]
                    ym = np.mean(y)
                    xs = xs + [xx-xm for xx in x]
                    ys = ys + [xx-ym for xx in y]
            try:
                F = lslinear(xs,ys)
            except RegressionError:
                F = {'slope':0,'intercept':-float('inf'),'r2':0}
            _Rt["t-"+roi] = Regressor(F['slope'],F['intercept'],F['r2'])
        self.roi_regressors = {**_Rs, **_Rt}
        ST.roi_regressors[self.subject] = self.roi_regressors


    def plot_roi_regressor(self,roi):
        fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(15,10))
        xs = []
        ys = []
        for i,xx in enumerate(self.rois):
            if xx==roi:continue
            ind = self._sts[(roi,xx)]
            can = self.data[ind]
            if (can!=roi and can.regressor.r2>75 and len(can.envelopes)>5):
                x = [can.data[xx][0] for xx in can.envelopes]
                xm = np.mean(x)
                y = [can.data[xx][1] for xx in can.envelopes]
                ym = np.mean(y)
                xs.append([xx-xm for xx in x])
                ys.append([xx-ym for xx in y])
        for v,w in zip(xs,ys):
            c=np.random.rand(3,)
            ax[0].plot(v,w,'s',ms=6,color=c)
        R = self.roi_regressors['s-'+roi]
        x = [item for sublist in xs for item in sublist]
        if R.r2>0:
            ax[0].plot(x,[R.predict(xx) for xx in x],'k-',lw=2)
            ax[0].text(20,0,f'R2={R.r2:.2f}, S={R.slope:.3f}',fontsize=8)
        else:
            ax[0].text(20,0,'No Regressor',fontsize=8)
        ax[0].set_title(f'{roi}:Source-fixed regressor',fontsize=18)
        ax[0].text(20,0,f'R2={R.r2:.2f}, S={R.slope:.3f}',fontsize=8)
        # target fixed
        xs = []
        ys = []
        for i,xx in enumerate(self.rois):
            if xx==roi:continue
            ind = self._sts[(xx,roi)]
            can = self.data[ind]
            if (can!=roi and can.regressor.r2>75 and len(can.envelopes)>5):
                x = [can.data[xx][0] for xx in can.envelopes]
                xm = np.mean(x)
                y = [can.data[xx][1] for xx in can.envelopes]
                ym = np.mean(y)
                xs.append([xx-xm for xx in x])
                ys.append([xx-ym for xx in y])
        for v,w in zip(xs,ys):
            c=np.random.rand(3,)
            ax[1].plot(v,w,'s',ms=6,color=c)
        R = self.roi_regressors['t-'+roi]
        x = [item for sublist in xs for item in sublist]
        if R.r2>0:
            ax[1].plot(x,[R.predict(xx) for xx in x],'k-',lw=2)
            ax[1].text(20,0,f'R2={R.r2:.2f}, S={R.slope:.3f}',fontsize=8)
        else:
            ax[1].text(20,0,'No Regressor',fontsize=8)
        ax[1].set_title(f'{roi}:Target-fixed regressor',fontsize=18)



    def is_connected(self, roi1, roi2, mania2=True):
        """Return whether a connection exists between the ROIs in the MANIA network generated

        :param roi1: Source ROI
        :type roi1: str
        :param roi2: Destination ROI
        :type roi2: str
        :param mania2: If True (default), MANIA2 connection is returned. If not, MANIA1 connection is returned
        :type mania2: bool
        :return: Whether a connection exists between ROI 1 and ROI 2 in the MANIA network
        """
        rois = sorted(self.rois)
        ind1, ind2 = rois.index(roi1), rois.index(roi2)
        if mania2:
            return bool(self.mania2_network[ind1, ind2])
        else:
            return bool(self.mania1_network[ind1, ind2])


    def plot_mania(self):
        _,den1,nar1,t1 = mania_on_mat(self.matrix1)
        _,den2,nar2,t2 = mania_on_mat(self.matrix2)
        fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(15,10))
        ax[0].plot(den1,nar1,'b-',lw=2,label='MANIA1')
        ax[0].plot(den2,nar2,'r-',lw=2,label='MANIA2')
        ax[0].set_xlabel('density',fontsize=18)
        ax[0].set_ylabel('NAR',fontsize=18)
        ax[1].plot(t1,den1,'b-',lw=2,label='MANIA1')
        ax[1].plot(t2,den2,'r-',lw=2,label='MANIA2')
        ax[1].set_xlabel('Threshold',fontsize=18)
        ax[1].set_ylabel('Density',fontsize=18)
        plt.legend()

    def describe(self):
        """
        Intended to print important metrics from the ensemble
        density, NAR, threshold, size
        """
        s = [f'Subject:{self.subject}',
             f'Number of ROIs:{len(self.rois)}',
             f'Number of STs:{len(self._sts)}',
             '--------------------',
             f'MANIA1 Results',
             f'Density:{density(self.mania1_network)}',
             f'NAR:{NAR(self.mania1_network)}',
             f'threshold:{self.threshold1}'
             '--------------------',
             f'MANIA2 Results',
             f'Density:{density(self.mania2_network)}',
             f'NAR:{NAR(self.mania2_network)}',
             f'threshold:{self.threshold2}'
             ]
        return '\n'.join(s)


#####
from multiprocessing import Pool


def compute_subject(subject, save=False):
    # print('Processing subject %s' % subject)
    sub = EnsembleST(['L' + str(i) for i in range(1, 181)], subject=subject)
    sub.preprocess()
    sub.noise_spectrum()
    sub.find_roi_regressors()
    sub.find_corrected_weights()
    sub.get_matrix1()
    sub.get_matrix2()
    sub.run_mania1()
    sub.run_mania2()
    if save:
        sub.save_to_db(run_id='No Reverse')
        update_roi_regressor(sub)
    return sub


def compute_subjects():
    print('p')
    subjects = [126426, 135124, 137431, 144125, 146735, 152427, 153227, 177140, 180533, 186545,
                188145, 192237, 206323, 227533, 248238, 360030, 361234, 362034, 368753, 401422,
                413934, 453542, 463040, 468050, 481042, 825654, 911849, 917558, 992673, 558960,
                569965, 644246, 654552, 680452, 701535, 804646, 814548]
    subjects = [126426, 137431, 144125, 146735, 152427, 135124, 192237, 206323, 227533, 248238,
                360030, 361234, 362034, 368753, 401422, 413934, 453542, 463040, 468050, 481042]
    subjects_h = [153227, 177140, 180533, 186545, 188145, 569965, 644246, 654552, 680452, 701535,
                804646, 814548, 825654, 911849, 917558, 992673, 558960]
    for subject in tqdm(subjects, desc='Sub'):
        compute_subject(subject, True)
