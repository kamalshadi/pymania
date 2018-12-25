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

    def __init__(self, subject, roi1, roi2, _length=None, _weight=None, border=0):
        self.subject = subject
        self.roi1 = roi1
        self.roi2 = roi2
        if _length is not None and _weight is not None:
            self.data = zip(_length, _weight)
        else:
            tmp = getdata_st(subject, roi1, roi2)
            self.data = zip(tmp['_length'], tmp['_weight'])
        if border > 0:
            self.border = border
        else:
            self.border = 0
        self._level = 0  # _level zero means no processing is yet done

    def __str__(self):
        if self.isNull():
            return f'S{self.subject}:{self.roi1}=>{self.roi2}:Null'
        return f'S{self.subject}:{self.roi1}=>{self.roi2}:{len(self)} seeds'

    def __repr__(self):
        return f'ST({self.subject},{self.roi1},{self.roi2})'

    @property
    def data(self):
        return self._data

    @property
    def weights(self):
        if self.isNull():
            return []
        return self.data[:, 1]

    @property
    def weight(self):
        if self.isNull():
            return np.log(1/NOS)
        return np.max(self.data[:, 1])

    @data.setter
    def data(self, vec):
        self._data = np.array(sorted([[xx[0], np.log(xx[1]/5000.0)] for xx in vec if xx[1] > 1]))

    @property
    def noise_threshold(self):
        if self._level > 0:
            return self._noise_threshold
        raise AttributeError('Please first run find_noise_threshold')

    @property
    def envelopes(self):
        if self.isNull():
            return []
        if self._level > 1:
            return self._envelopes
        raise AttributeError('Please first run find_envelope_points')

    @property
    def regressor(self):
        if self.isNull():
            return Regressor(0, np.log(1/NOS), 0)
        if self._level > 2:
            return self._regressor
        raise AttributeError('Please first run find_local_regressor')

    @regressor.setter
    def regressor(self,dic):
        self._regressor = Regressor(dic['slope'],dic['intercept'],dic['r2'])

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

    @property
    def corrected_weights(self):
        if self._level>4:
            return self._corrected_weights
        raise MANIA2Error(f'Weights are not yet corrected for {self.roi1} to {self.roi2}')

    @corrected_weights.setter
    def corrected_weights(self,arg):
        self._corrected_weights = arg
        self._level = 5

    @property
    def corrected_weight(self):
        if self._level>4:
            if self.isNull():
                return np.log(1/NOS)
            if len(self.corrected_weights)>0:
                return np.median(self.corrected_weights)
            return np.log(1/NOS)
        raise MANIA2Error(f'Weights are not yet corrected for {self.roi1} to {self.roi2}')

    def __len__(self):
        return len(self.data)


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



    def find_envelope_points(self):
        outs = []
        l = len(self)
        for i,cur in enumerate(self.data):
            if i >= (l - MIN_POINTS_ON_RIGHT):
                break
            if noise_threshold<0:
                if self.data[i][1]<noise_threshold:
                    continue
            else:
                if self.data[i][1] < noise_threshold:
                    continue
            right_side = self.data[(i+1):,:]
            tmp = right_side.max(axis=0)[1]
            if self.data[i][1] > tmp:
                outs.append(i)
        self._envelopes = outs
        self._level = 2
        return outs

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
            return self.border>0
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
        The index of the seed with maxim weight to from the source to target.
        '''
        return np.argmax(self.data,axis=0)[1]

    def find_local_regressor(self):

        if self.isNull():
            raise MANIA2Error('Connection is null')
        if self._level<2:
            raise MANIA2Error('Please first run find_envelope_points')

        try:
            self.regressor = lslinear(self.data[self.envelopes,0],self.data[self.envelopes,1])
        except RegressionError as e:
            if verbose:
                print(e)
            self.regressor = {'slope':0,'intercept':self.max()[1],'r2':0}
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
        ax.axhline(noise_threshold,ls='--',color='black',lw=2,label="Global noise threshold")
        if self._level>0:
            ax.axhline(self.noise_threshold,ls='--',color='r',lw=2,label="Noise threshold")
        if self._level>1:
            ax.plot(self.data[self.envelopes,0],self.data[self.envelopes,1],'gs',ms=6,label="Envelope points")
        if self._level>2:
            x = self.data[self.envelopes,0]
            z = list(map(self.regressor.predict,x))
            ax.plot(x,z,'k',lw=2,label='Local regressor')
            try:
                R = ST.roi_regressors[self.subject][f's-{self.roi1}']
                if len(x)>0:
                    c = R.correct([x[-1],z[-1]])
                else:
                    t = self.max()
                    x = [t[0]]
                    z = [t[1]]
                    c = R.correct([x[-1],z[-1]])
                ax.plot([0,x[-1]],[c,z[-1]],'r--',lw=2,label='SF regressor')
            except KeyError:
                pass
            try:
                R = ST.roi_regressors[self.subject][f't-{self.roi2}']
                if len(x)>0:
                    c = R.correct([x[-1],z[-1]])
                else:
                    t = self.max()
                    x = [t[0]]
                    z = [t[1]]
                    c = R.correct([x[-1],z[-1]])
                ax.plot([0,x[-1]],[c,z[-1]],'m--',lw=2,label='SF regressor')
            except KeyError:
                pass
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
    '''
    EnsembleST is mainly designed to contain all STs for a single subject study.
    In future, we may extend this class to contain a single ST
    from across a subject cohort.
    '''
    def __init__(self,arg,**kwargs):
        try:
            self.subject = kwargs['subject']
            self.mode = 0 # single subject mode
        except KeyError:
            self._mode = 1
        self.data = arg
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
        elif isinstance(tmp,PairST):
            self._data = [tmp.st1]*2*len(arg)
            self._sts = {}
            self._rois = set([])
            for i,pair in enumerate(arg):
                self._data[2*i] = pair.st1
                self._data[2*i+1] = pair.st2
                self._sts[(pair.st1.roi1,pair.st1.roi2)] = 2*i
                self._sts[(pair.st2.roi1,pair.st2.roi2)] = 2*i + 1
                self._rois.add(pair.st1.roi1)
                self._rois.add(pair.st2.roi1)
                self._rois.add(pair.st1.roi2)
                self._rois.add(pair.st2.roi2)
        elif isinstance(tmp,str):
            assert self.mode==0, 'API only works in a single subject mode - Please specify subject ID -> (subject=ID)'
            data = getdata_sts(self.subject,arg)
            self._sts = {(xx['n1'],xx['n2']):i for i,xx in enumerate(data)}
            self._data = [ST(self.subject,*xx.values()) for xx in data]
            self._rois = set(arg)
        else:
            raise ValueError('Ensemble constructor arguments not known!')

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

    def load_roi_regressors(self,fp):
        with open(fp,'rb') as f:
            D = pk.load(f)
        self.roi_regressors = D
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


    def find_corrected_weights(self):
        for conn in self.data:
            if conn.isNull():
                conn.corrected_weights = []
                continue
            if len(conn.envelopes)>0:
                envs = conn.data[conn.envelopes,:]
            elif conn.max()[1]>noise_threshold:
                # no envelope point but above noise points
                envs = conn.data[conn.argmax(),:].reshape(1,-1)
            else:
                conn.corrected_weights = []
                continue
            Rs = self.roi_regressors['s-'+conn.roi1]
            Rt = self.roi_regressors['t-'+conn.roi2]
            tmp = list(map(Rs.correct,envs))+list(map(Rt.correct,envs))
            conn.corrected_weights = tmp

    def get_matrix1(self):
        '''
        Matrix1 elements are based on the strongest connected seed between two
        terminal ROIs
        '''
        try:
            return self.matrix1
        except AttributeError:
            pass
        l = len(self.rois)
        mat = np.zeros((l,l))
        rois = sorted(self.rois)
        for i,roi1 in enumerate(rois):
            for j,roi2 in enumerate(rois):
                if roi1==roi2:continue
                ind = self._sts[(roi1,roi2)]
                conn = self.data[ind]
                mat[i,j] = np.exp(conn.weight)*NOS
        self.matrix1 = mat
        return mat


    def get_matrix2(self):
        '''
        Matrix2 elements are corrected by our distance correction framework
        '''
        try:
            return self.matrix2
        except AttributeError:
            pass
        l = len(self.rois)
        mat = np.zeros((l,l))
        rois = sorted(self.rois)

        # Different modes of correction applied to pairST
        for i,roi1 in enumerate(rois[:l-1]):
            for j,roi2 in enumerate(rois[i+1:]):
                ind = self._sts[(roi1,roi2)]
                ind_reverse = self._sts[(roi2,roi1)]
                conn = self.data[ind]
                conn_reverse = self.data[ind_reverse]

                # check if a direction is null -> no correction
                if conn.isNull() or conn_reverse.isNull():
                    mat[i,j+i+1] = np.exp(conn.weight)*NOS
                    mat[j+i+1,i] = np.exp(conn_reverse.weight)*NOS
                    conn.correction_type = 'null'
                    conn_reverse.correction_type = 'null'
                    continue

                # check if a direction is strongly adjacent -> no correction
                if (conn.isAdjacent(True) or conn_reverse.isAdjacent(True)):
                    mat[i,j+i+1] = np.exp(conn.weight)*NOS
                    mat[j+i+1,i] = np.exp(conn_reverse.weight)*NOS
                    conn.correction_type = 'strongly adjacent'
                    conn_reverse.correction_type = 'strongly adjacent'
                    continue

                # check if both have envelope points -> correction applied
                if (len(conn.envelopes)>0 and len(conn_reverse.envelopes)>0):
                    mat[i,j+i+1] = np.exp(min(conn.corrected_weight,0))*NOS
                    mat[j+i+1,i] = np.exp(min(conn_reverse.corrected_weight,0))*NOS
                    conn.correction_type = 'envelope'
                    conn_reverse.correction_type = 'envelope'
                    continue

                # check if both are above noise -> correction applied
                if (conn.max()[1]>noise_threshold and conn_reverse.max()[1]>noise_threshold):
                    mat[i,j+i+1] = np.exp(min(conn.corrected_weight,0))*NOS
                    mat[j+i+1,i] = np.exp(min(conn_reverse.corrected_weight,0))*NOS
                    conn.correction_type = 'above noise'
                    conn_reverse.correction_type = 'above noise'
                    continue
                else:
                    # fallback no correction
                    mat[i,j+i+1] = np.exp(conn.weight)*NOS
                    mat[j+i+1,i] = np.exp(conn_reverse.weight)*NOS
                    conn.correction_type = 'fallback'
                    conn_reverse.correction_type = 'fallback'
        self.matrix2 = mat
        return mat

    def run_mania1(self):
        '''
        Running MANIA on matrix1
        '''
        net,den,nar,t = mania_on_mat(self.matrix1)
        self.mania1_network = net

    def run_mania2(self):
        '''
        Running MANIA on matrix2
        '''
        net,den,nar,t = mania_on_mat(self.matrix2)
        self.mania2_network = net

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

    def save_to_db(self):
        """Save the connections between all the ROIs to Neo4j database

        :return: None
        """
        rois = sorted(self.rois)
        for roi1 in tqdm(rois, desc='ROIs'):
            for roi2 in rois:
                if roi1 == roi2: continue
                ind = self._sts[(roi1, roi2)]
                conn = self.data[ind]
                attributes = {'SUBJECT':self.subject,
                              'corrected_weight': conn.corrected_weight,
                              'corrected_weights': conn.corrected_weights,
                              'correction_type': conn.correction_type,
                              'noise_threshold': conn.noise_threshold,
                              'is_adjacent': conn.isAdjacent(True),
                              'is_connected': self.is_connected(roi1, roi2),
                              'is_connected_mania1': self.is_connected(roi1, roi2, mania2=False),
                              'regressor': conn.regressor.to_list(),
                              'envelope': conn.envelopes,
                              'weight': conn.weight,
                              'weights': conn.weights
                              }
                write_connection(roi1, roi2, 'MANIA2', attributes)

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
             '--------------------',
             f'MANIA2 Results',
             f'Density:{density(self.mania2_network)}',
             f'NAR:{NAR(self.mania2_network)}']
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
        sub.save_to_db()
        update_roi_regressor(sub)
    # print('Completed subject %s' % subject)
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
