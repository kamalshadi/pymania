# rank aggregation used by group mania
import numpy as num
from collections import defaultdict

def ensemble_order(L,shuffle=True):
    D = defaultdict(int)
    l = len(L)
    for sub in L:
        for pair in sub:
            D[(pair[0],pair[1])] += pair[-1]
    out = [(roi[0],roi[1],wei//l) for roi,wei in sorted(D.items(),reverse=True, key=lambda x:x[-1])]
    if shuffle:
        num.random.shuffle(out)
    return out

def KendalMatrix(L,roiIndex):
    # L is the list of list
    # Each sublist is the rank of edges bu a subject
    # Each tuple is (roi1,roi2,weight)
    # roiIndex is the dictionry mapping roi to index in connectivity matrix
    m=len(L[0]) # number of pairs
    S = len(L)
    W=num.zeros((m,m),dtype='uint8')
    nroi = len(roiIndex)
    for q,sub in enumerate(L):
        print(q+1,S)
        cohort = set([])
        his = sub[0][-1]
        so_far = set([])
        for q,pair in enumerate(sub):
            if his<=10:
                break
            roi1 = pair[0]
            roi2 = pair[1]
            r1 = roiIndex[roi1]
            r2 = roiIndex[roi2]
            index = r1*nroi + r2
            if r2<r1:
                index -= r1
            elif r2>r1:
                index -= (r1+1)
            else:
                raise Exception('Unexpected Indexing....')
            if pair[-1]==his and q<(S-1):
                cohort.add(index)
                so_far.add(index)
                continue
            W[list(cohort),:] +=1
            x,y = num.meshgrid(list(cohort), list(so_far), sparse=True, indexing='ij')
            W[x,y] -= 1
            so_far.add(index)
            cohort = set([index])
            his = pair[-1]
    W[range(m),range(m)]=0
    return W





def agg(W,R,roiIndex):
    # start R with a random ranking
    l=len(R)
    nroi = len(roiIndex)
    if l==2:
        _i = roiIndex[R[0][0]]
        _j = roiIndex[R[0][1]]
        i1 = _i*len(roiIndex)+_j
        if _j<_i:
            i1 -= _i
        elif _j>_i:
            i1 -= (_i+1)
        else:
            raise Exception('Unexpected Indexing....')
        _i = roiIndex[R[1][0]]
        _j = roiIndex[R[1][1]]
        i2 = _i*len(roiIndex)+_j
        if _j<_i:
            i2 -= _i
        elif _j>_i:
            i2 -= (_i+1)
        else:
            raise Exception('Unexpected Indexing....')
        if W[i1,i2]>W[i2,i1]:
            return R
        else:
            return [R[1],R[0]]
    elif l<2:
        return R
    else:
        piv=num.random.randint(0,l)
        Rl=[]
        Rr=[]
        _i = roiIndex[R[piv][0]]
        _j = roiIndex[R[piv][1]]
        i2 = _i*nroi+_j
        if _j<_i:
            i2 -= _i
        elif _j>_i:
            i2 -= (_i+1)
        else:
            raise Exception('Unexpected Indexing....')
        for i in range(l):
            if i==piv:continue
            _i = roiIndex[R[i][0]]
            _j = roiIndex[R[i][1]]
            i1 = _i*len(roiIndex)+_j
            if _j<_i:
                i1 -= _i
            elif _j>_i:
                i1 -= (_i+1)
            else:
                raise Exception('Unexpected Indexing....')
            if W[i1,i2]>W[i2,i1]:
                Rl.append(R[i])
            elif W[i1,i2]<W[i2,i1]:
                Rr.append(R[i])
            else:
                if num.random.rand()<.5:
                    Rl.append(R[i])
                else:
                    Rr.append(R[i])
        return agg(W,Rl,roiIndex)+[R[piv]]+agg(W,Rr,roiIndex)
