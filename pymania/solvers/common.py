from ..config import *
import numpy as np
from ..utils import *
from collections import deque
from ..primitives import *


def find_noise_threshold(subject=None,st=None):
    if subject is None and st is None:
        return np.log(1/config.NOS)
    raise Exception('Noise threshold detection not implemented!')

def find_envelope_points(arg,noise_threshold):
    if isinstance(arg,ST):
        st = arg
        '''st is Source-Target primitive'''
        outs = []
        l = len(st)
        for i,cur in enumerate(st):
            if i >= (l - config.MIN_POINTS_ON_RIGHT):
                break
            if st.data[i][1]<noise_threshold:
                continue
            right_side = st.data[(i+1):,:]
            tmp = right_side.max(axis=0)[1]
            if st.data[i][1] > tmp:
                outs.append(i)
        st._envelopes = outs
    else:
        st1 = arg.st1
        st2 = arg.st2
        if st1.isNull() or st2.isNull():
            find_envelope_points(st2,np.log(1/config.NOS))
            find_envelope_points(st1,np.log(1/config.NOS))
            return
        l1 = len(st1)
        l = l1 + len(st2)
        merged = [[0,0,1] for i in range(l)]
        for i,elem in enumerate(st1):
            merged[i][0] = elem[0]
            merged[i][1] = elem[1]
        for i,elem in enumerate(st2):
            merged[i+l1][0] = elem[0]
            merged[i+l1][1] = elem[1]
            merged[i+l1][2] = 2
        merged = np.array(sorted(merged))
        outs = []
        i1,i2 = 0,0
        for i,cur in enumerate(merged):
            if cur[2]==1:
                i1 += 1
            else:
                i2 += 1
            if i >= (l - config.MIN_POINTS_ON_RIGHT):
                break
            if cur[1]<noise_threshold:
                continue
            right_side = merged[(i+1):,:]
            tmp = right_side.max(axis=0)[1]
            if cur[2]==1:
                if cur[1]>tmp:
                    outs.append((i1-1,1))
            else:
                if cur[1]>tmp:
                    outs.append((i2-1,2))

        env1 = []
        env2 = []
        for k,w in outs:
            if w==1:
                env1.append(k)
            else:
                env2.append(k)
        st1._envelopes = env1
        st2._envelopes = env2


def find_local_regressor(arg):
    if isinstance(arg,ST):
        st = arg
        if st.isNull():
            tmp = np.log(1/config.NOS)
            st.regressor = {'slope':0,'intercept':tmp,'r2':0}
            return
        try:
            st.regressor = lslinear(st.data[st.envelopes,0],st.data[st.envelopes,1])
        except RegressionError as e:
            st.regressor = {'slope':0,'intercept':st.max()[1],'r2':0}
    else:
        st1 = arg.st1
        st2 = arg.st2
        if st1.isNull() and st2.isNull():
            tmp = np.log(1/config.NOS)
            st1.regressor = {'slope':0,'intercept':tmp,'r2':0}
            st2._regressor = st1.regressor
            return
        elif st1.isNull():
            find_local_regressor(st2)
            st1._regressor = st2.regressor
            return
        elif st2.isNull():
            find_local_regressor(st1)
            st2._regressor = st1.regressor
            return
        else:
            x = list(st1.data[st1.envelopes,0])+list(st2.data[st2.envelopes,0])
            y = list(st1.data[st1.envelopes,1])+list(st2.data[st2.envelopes,1])
            try:
                st1.regressor = lslinear(x,y)
                st2._regressor = st1.regressor
            except RegressionError as e:
                st1.regressor = {'slope':0,'intercept':st1.max()[1],'r2':0}
                st2.regressor = {'slope':0,'intercept':st2.max()[1],'r2':0}



def find_corrected_weights(st):
    if st.isNull():
        st._corrected_weights = []
        st._corrected_weight = np.log(1/config.NOS)
        st.correction_type = 'Null'
        return
    if st.isAdjacent(True):
        st._corrected_weights = [st.max()[1]]
        st._corrected_weight = st.max()[1]
        st.correction_type = 'Adjacent'
        return
    if len(st.envelopes)>0:
        if st.regressor.r2>=config.MIN_R2:
            st.correction_type = 'Regress'
            envs = st.data[st.envelopes,:]
            tmp = list(map(st.regressor.correct,envs))
        else:
            st.correction_type = 'Bad regressor'
            envs = st.data[st.envelopes,:]
            tmp = [st.max()[1]]

    else:
        st.correction_type = 'No Envelope'
        tmp = [st.max()[1]]
    st._corrected_weights = tmp
    st._corrected_weight = np.median(tmp)
