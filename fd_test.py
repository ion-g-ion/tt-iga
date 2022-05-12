import torch as tn
import torchtt as tntt
import numpy as np
import matplotlib.pyplot as plt
import timeit
import scipy
import scipy.sparse 
import scipy.sparse.linalg

tn.set_default_dtype(tn.float64)

size_tt = lambda T : sum([c.size for c in T.to_list(T)])

# def custom_meshgri
# u = lambda x,y,z : 1/(1+x**2+y**2+z**2)+np.sin(x*y)
# f = lambda x,y,z : 2*(x**2+y**2+z**2-3)/(x**2+y**2+z**2+1)**3  - (y**2+x**2)*np.sin(x*y)

n = 4
h = 1/(n-1)

l= 1
X1, Y1, Z1 = tn.meshgrid(tn.linspace(0,l,n) ,tn.linspace(0,l,n),tn.linspace(0,l,n),indexing='ij')
X2, Y2, Z2 = tn.meshgrid(tn.linspace(-l,0,n),tn.linspace(0,l,n),tn.linspace(0,l,n),indexing='ij')
X3, Y3, Z3 = tn.meshgrid(tn.linspace(-l,0,n),tn.linspace(-l,0,n),tn.linspace(0,l,n),indexing='ij')

f1_tt = tntt.ones([n,n,n])
f2_tt = tntt.ones([n,n,n])
f3_tt = tntt.ones([n,n,n])
f_tt = f1_tt ** tntt.TT(tn.tensor([1.0,0,0])) + f2_tt ** tntt.TT(tn.tensor([0.0,1,0])) + f3_tt ** tntt.TT(tn.tensor([0,0,1.0]))
f_tt = f_tt.round(1e-12)

# ur1_tt = tt.tensor(u(X1,Y1,Z1))
# ur2_tt = tt.tensor(u(X2,Y2,Z2))
# ur3_tt = tt.tensor(u(X3,Y3,Z3))
# ur_tt = tt.kron(ur1_tt,tt.tensor(np.array([1,0,0]))) + tt.kron(ur2_tt,tt.tensor(np.array([0,1,0]))) + tt.kron(ur3_tt,tt.tensor(np.array([0,0,1])))
# ur_tt = ur_tt.round(1e-12)

g_tt = tntt.ones([n,n,n,3]) * 0



ID1 = tn.eye(n)
ID1[0,0] = 0
ID1[-1,-1] = 0
ID2 = tn.zeros((n,n))
ID2[0,0] = 1
Pin1_tt = tntt.rank1TT([ID1, ID1, ID1]) + tntt.rank1TT([ID2, ID1, ID1])

ID1 = tn.eye(n)
ID1[0,0] = 0
ID1[-1,-1] = 0
ID2 = tn.zeros((n,n))
ID2[-1,-1] = 1
ID3 = tn.zeros((n,n))
ID3[0,0] = 1
Pin2_tt = tntt.rank1TT([ID1, ID1, ID1])
Pin2_tt += tntt.rank1TT([ID2, ID1, ID1])
Pin2_tt += tntt.rank1TT([ID1, ID3, ID1])

ID1 = tn.eye(n)
ID1[0,0] = 0
ID1[-1,-1] = 0
ID2 = tn.zeros((n,n))
ID2[-1,-1] = 1
Pin3_tt = tntt.rank1TT([ID1, ID1, ID1]) + tntt.rank1TT([ID1, ID2, ID1])

Pin_tt = Pin1_tt ** tntt.diag(tntt.TT(tn.tensor([1.0,0,0]))) + Pin2_tt ** tntt.diag(tntt.TT(tn.tensor([0.0,1.0,0]))) + Pin3_tt ** tntt.diag(tntt.TT(tn.tensor([.0,0,1])))
Pbd_tt = tntt.eye([n,n,n,3]) - Pin_tt
# Pin_tt = Pin_tt.round(1e-12)


# f_tt = tt.ones([n,n,n,3])

# plt.figure()
# plt.contourf(X1[:,:,2],Y1[:,:,2],tt.matvec(Pin_tt,f_tt).full()[:,:,2,0])
# plt.contourf(X2[:,:,2],Y2[:,:,2],tt.matvec(Pin_tt,f_tt).full()[:,:,2,1])
# plt.contourf(X3[:,:,2],Y3[:,:,2],tt.matvec(Pin_tt,f_tt).full()[:,:,2,2])
# plt.colorbar()


L1d = -2*tn.eye(n)+tn.diag(tn.ones((n-1)),-1)+tn.diag(tn.ones((n-1)),1)
L1d /= h**2
# L1d[0,:] = 0
# L1d[-1,:] = 0
ID = tn.eye(n)
# ID[0,:] = 0 
# ID[-1,:] = 0

L_tt =  tntt.rank1TT([L1d, ID, ID])
L_tt += tntt.rank1TT([ID, L1d, ID])
L_tt += tntt.rank1TT([ID, ID, L1d])
L_tt = (Pin1_tt@L_tt) ** tntt.diag(tntt.TT(tn.tensor([1.0,0,0]))) + (Pin2_tt@L_tt) ** tntt.diag(tntt.TT(tn.tensor([0.0,1.0,0]))) + (Pin3_tt@L_tt) ** tntt.diag(tntt.TT(tn.tensor([.0,0,1])))


# interfacing
L1d = -2*tn.eye(n)+tn.diag(tn.ones((n-1)),-1)+tn.diag(tn.ones((n-1)),1)
L1d /= h**2
tmp = L1d.clone()
tmp[1:,:] = 0
tmp1_tt = tntt.rank1TT([tmp, ID, ID])
tmp = L1d.clone()
tmp[:-1,:] = 0
tmp2_tt = tntt.rank1TT([tmp, ID, ID])

# L_tt += tt.kron(tmp1_tt,tt.matrix(np.diag(np.array([1,0,0]))))+tt.kron(tmp2_tt,tt.matrix(np.diag(np.array([0,1,0]))))

zz = tn.zeros((n,n))
zz[0,-2] = 1
tmp = ID.clone()
tmp[0,:] = 0
tmp[-1,:] = 0
L12_tt = tntt.rank1TT([zz,tmp,tmp, tn.tensor([[0.0,1,0],[0,0,0],[0,0,0]])])
L_tt += L12_tt*(1/h/h)
zz = tn.zeros((n,n))
zz[-1,1] = 1
L21_tt = tntt.rank1TT([zz, tmp, tmp,tn.tensor([[0,0,0],[1,0,0],[0,0,0]])])
L_tt += L21_tt*(1/h/h)

zz = tn.zeros((n,n))
zz[0,-2] = 1
tmp = tn.eye(n)
tmp[0,:] = 0
tmp[-1,:] = 0
L23_tt = tntt.rank1TT([tmp,zz,tmp, tn.tensor([[0.0,0,0],[0,0,1],[0,0,0]])])     
L_tt += L23_tt*(1/h/h)
zz = tn.zeros((n,n))
zz[-1,1] = 1
L32_tt = tntt.rank1TT([tmp,zz,tmp, tn.tensor([[0.0,0,0],[0,0,1],[0,0,0]]).t()])     
L_tt += L32_tt*(1/h/h)


# plt.figure()
# plt.contourf(X1[:,:,2],Y1[:,:,2],tt.matvec(L_tt,ur_tt).full()[:,:,2,0])
# plt.contourf(X2[:,:,2],Y2[:,:,2],tt.matvec(L_tt,ur_tt).full()[:,:,2,1])
# plt.contourf(X3[:,:,2],Y3[:,:,2],tt.matvec(L_tt,ur_tt).full()[:,:,2,2])
# plt.colorbar()

Pbd_tt = Pbd_tt*(-1/h/h)
rhs_tt = Pin_tt @ f_tt + Pbd_tt @ g_tt
rhs_tt = rhs_tt.round(1e-12)

M_tt = L_tt+Pbd_tt
M_tt = M_tt.round(1e-12)

x0 = rhs_tt.round(1e-12,4)

tme_amen = timeit.time.time()
u_tt = tntt.solvers.amen_solve(M_tt, rhs_tt, x0 = x0, eps = 1e-6,nswp = 40)
tme_amen = timeit.time.time() - tme_amen

plt.figure()
plt.contourf(X1[:,:,2],Y1[:,:,2],u_tt[:,:,2,0].numpy(),levels=64,vmin = -0.04,vmax = 0)
plt.contourf(X2[:,:,2],Y2[:,:,2],u_tt[:,:,2,1].numpy(),levels=64,vmin = -0.04, vmax = 0)
plt.contourf(X3[:,:,2],Y3[:,:,2],u_tt[:,:,2,2].numpy(),levels=64, vmin = -0.04,vmax = 0)
plt.colorbar()

# plt.figure()
# plt.contourf(X1[:,:,2],Y1[:,:,2],ur_tt[:,:,2,0].full(),levels=64)
# plt.contourf(X2[:,:,2],Y2[:,:,2],ur_tt[:,:,2,1].full(),levels=64)
# plt.contourf(X3[:,:,2],Y3[:,:,2],ur_tt[:,:,2,2].full(),levels=64)
# plt.colorbar()

# plt.figure()
# plt.contourf(X1[:,:,2],Y1[:,:,2],np.log10(np.abs(u_tt[:,:,2,0].full()-ur_tt[:,:,2,0].full())),levels=64)
# plt.contourf(X2[:,:,2],Y2[:,:,2],np.log10(np.abs(u_tt[:,:,2,1].full()-ur_tt[:,:,2,1].full())),levels=64)
# plt.contourf(X3[:,:,2],Y3[:,:,2],np.log10(np.abs(u_tt[:,:,2,2].full()-ur_tt[:,:,2,2].full())),levels=64)
# plt.colorbar()

# err2 = (u_tt-ur_tt).norm()/ur_tt.norm()

print('{0:32s} : {1:12f}'.format("Time AMEN [s]",tme_amen))

# print('{0:32s} : {1:12e}'.format("l2 relative",err2))

#%% Reference
import sys
sys.exit()
M_full = M_tt.numpy().reshape([n**3 *3 ,-1])
M_full[np.abs(M_full)<1e-8] = 0
M_sp = scipy.sparse.csr_matrix(M_full)
rhs_full = rhs_tt.numpy().transpose().reshape([-1,1])

tme_solver = timeit.time.time()
u_full = scipy.sparse.linalg.spsolve(M_sp,rhs_full).reshape([3,n,n,n]).transpose()
tme_solver = timeit.time.time() - tme_solver
print('{0:32s} : {1:12f}'.format("Time scipy solver [s]",tme_solver))

plt.figure()
plt.contourf(X1[:,:,2],Y1[:,:,2],u_full[:,:,2,0],levels=64)
plt.contourf(X2[:,:,2],Y2[:,:,2],u_full[:,:,2,1],levels=64)
plt.contourf(X3[:,:,2],Y3[:,:,2],u_full[:,:,2,2],levels=64)
plt.colorbar()

print('{0:32s} : {1:12f}'.format("Compression ratio",tntt.numel(u_tt.round(1e-8))/u_full.size))
