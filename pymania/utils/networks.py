import numpy as np

def mania_on_mat(B,nos = 5000,cut = 10, log=True):
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
		if tmp<his:
			his = tmp
			net = C
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
