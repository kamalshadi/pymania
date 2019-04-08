import numpy as np

def mania_on_mat(B,nos = 5000,cut = 10, log=False):
	l = nos-2*cut
	den = [0.0]*l
	nar = [0.0]*l
	t = [0.0]*l
	n,_ = B.shape
	his = float('inf')
	C = np.zeros((n,n))
	for i in range(nos-cut,cut,-1):
		if log:
			th = np.log(i/nos)
		else:
			th = i
		t[i-cut-1] = th
		C[B>=th] = 1
		den[i-cut-1] = density(C)
		tmp = NAR(C)
		nar[i-cut-1] = tmp
	ind = np.argmin(nar)
	th = t[ind]
	net = np.zeros((n,n))
	net[B>=th] = 1
	return (net,den,nar,t)

def mania_on_rank(R,roi2ind,nos = 5000,cut=10):
	m = len(R)
	N = int(np.round(0.5*(1+(4*m)**.5)))
	net = np.zeros((N,N))
	den = [0.0]*m
	nar = [np.nan]*m
	t = [np.nan]*m
	delta = 1/m
	P0 = min(R[0][-1],5000)
	for q,tup in enumerate(R):
		roi1 = tup[0]
		roi2 = tup[1]
		i = roi2ind[roi1]
		j = roi2ind[roi2]
		weight = tup[-1]
		net[i,j] = 1
		den[q] += q*delta
		if q<cut or q>m-cut or weight>=P0:
			continue
		t[q] = tup[-1]
		nar[q] = NAR(net)
	ind = np.nanargmin(nar)+1
	net = np.zeros((N,N))
	for q,tup in enumerate(R[:ind]):
		roi1 = tup[0]
		roi2 = tup[1]
		i = roi2ind[roi1]
		j = roi2ind[roi2]
		net[i,j] = 1
	return (net,den,nar,t)



def AR(A):
	# Computes Assymetry ratio
	B=A-A.transpose()
	total_edges=sum(sum(A-np.diag(np.diag(A))))
	eu=float(sum(sum(abs(B))))/2
	if total_edges==0:
		return 1.0
	return eu/total_edges

def NAR(A,factor = 1.0):
	# Computes the normalized asymmetry ratio
	l1,l2=np.shape(A)
	if l1!=l2:
		print('Error: Input matrix not square')
		return None
	l=l1
	total_edges=sum(sum(A-np.diag(np.diag(A))))
	AR_rand=1-(float(factor*total_edges)/(l*(l-1)))
	if AR_rand==0:
		return float('inf')
	return AR(A)/AR_rand

def sim(A1,A2,mode='jac'):
	# Jaccard similarity on edges
	l1,l2=np.shape(A1)
	if l1!=l2:
		print('Error: Input matrix not square')
		return None
	l=l1
	l1,l2=np.shape(A1)
	if l1!=l or l2!=l:
		print('Error: Matrices must be of the same size')
		return None
	if mode=='jac':
		P=A1+A2
		P=P-np.diag(np.diag(P))
		P[P>0]=1
		S=A1*A2
		S=S-np.diag(np.diag(S))
		return float(sum(sum(S)))/sum(sum(P))
	else:
		S=A1*A2
		S=S-np.diag(np.diag(S))
		A1=A1-np.diag(np.diag(A1))
		A2=A2-np.diag(np.diag(A2))
		denom=min(sum(sum(A1)),sum(sum(A2)))
		return float(sum(sum(S)))/denom

def density(A):
	# returns the density of edges in A
	l1,l2=np.shape(A)
	if l1!=l2:
		print('Error: Input matrix not square')
		return None
	l=l1
	total_edges=sum(sum(A-np.diag(np.diag(A))))
	return float(total_edges)/(l*(l-1))
