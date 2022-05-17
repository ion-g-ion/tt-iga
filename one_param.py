import torch as tn
import torchtt as tntt
import matplotlib.pyplot as plt
import tt_iga
import numpy as np
import datetime
import scipy.sparse
import scipy.sparse.linalg
import pandas as pd

tn.set_default_dtype(tn.float64)

   
Ns = [100]*3
deg = 2
nl = 8
alpha = 1/4 
eps_solver = 1e-8
eps_construction = 1e-11

baza1 = tt_iga.bspline.BSplineBasis(np.linspace(0,1,Ns[0]-deg+1),deg)
baza2 = tt_iga.bspline.BSplineBasis(np.linspace(0,1,Ns[1]-deg+1),deg)
baza3 = tt_iga.bspline.BSplineBasis(np.linspace(0,1,Ns[2]-deg+1),deg)

Basis = [baza1,baza2,baza3]
N = [baza1.N,baza2.N,baza3.N]

Basis_param = [tt_iga.lagrange.LagrangeLeg(nl,[0,1])]

xc = lambda u,v: u*np.sqrt(1-v**2/2)
yc = lambda u,v: v*np.sqrt(1-u**2/2)

xparam = lambda t : xc(t[:,0]*2-1,t[:,1]*2-1)*((1+np.cos((t[:,2]*2-1)*np.pi))*alpha*t[:,3]+1)
yparam = lambda t : yc(t[:,0]*2-1,t[:,1]*2-1)*((1+np.cos((t[:,2]*2-1)*np.pi))*alpha*t[:,3]+1)
zparam = lambda t : t[:,2]*2-1

geom = tt_iga.PatchBSpline.interpolate_geometry([xparam, yparam, zparam], Basis, Basis_param)

tme = datetime.datetime.now() 
Mass_tt = geom.mass_interp(Basis, eps=1e-12)
tme = datetime.datetime.now() -tme
print('Time mass matrix ',tme.total_seconds())
tme_mass = tme.total_seconds()


tme = datetime.datetime.now() 
Stiff_tt = geom.stiffness_interp(Basis, eps = eps_construction, qtt = False, verb=True)
tme = datetime.datetime.now() -tme
print('Time stiffness matrix ',tme.total_seconds())
tme_stiff = tme.total_seconds()

N = [baza1.N,baza2.N,baza3.N]


# interpolate rhs and reference solution
sigma = 0.5
uref = lambda x: np.exp(-((x[:,0]-0.0)**2+(x[:,1]-0.0)**2+(x[:,2]-0)**2)/sigma)
ffun = lambda x: -np.exp(-((x[:,0]-0.0)**2+(x[:,1]-0.0)**2+(x[:,2]-0)**2)/sigma)*(-6*sigma+4*((x[:,0]-0.0)**2+(x[:,1]-0.0)**2+(x[:,2]-0)**2))/sigma/sigma

kx = 2
ky = 3
uref = lambda x: np.sin(kx*x[:,0])*np.cos(ky*x[:,1])*np.exp(-np.sqrt(kx*kx+ky*ky)*x[:,2])
ffun = lambda x: x[:,0]*0

uref_fun = tt_iga.Function(Basis+Basis_param)
uref_fun.interpolate(uref, geometry = geom, eps = 1e-14)

f_fun = tt_iga.Function(Basis+Basis_param)
f_fun.dofs = tntt.zeros(uref_fun.dofs.N)


Pin_tt, Pbd_tt = tt_iga.projectors.get_projectors(N,[[0,0],[0,0],[0,0]]) 


Pin_tt = Pin_tt ** tntt.eye(Stiff_tt.N[3:])
Pbd_tt = Pbd_tt ** tntt.eye(Stiff_tt.N[3:])


Pbd_tt = (N[0]**-1) * Pbd_tt
M_tt = (Pin_tt@Stiff_tt+Pbd_tt).round(eps_construction)
rhs_tt = (Pin_tt @ Mass_tt @ f_fun.dofs + Pbd_tt @ uref_fun.dofs ).round(eps_construction)

print('System matrix... ',flush=True)


print('Rank Mtt ',M_tt.R)
print('Rank rhstt ',rhs_tt.R)
print('Rank uref TT',uref_fun.dofs.R)

tme = datetime.datetime.now() 
print('eps solver ',eps_solver,flush=True)
# dofs_tt = tntt.solvers.amen_solve(M_tt.cuda(), rhs_tt.cuda(), x0 = tntt.ones(rhs_tt.N).cuda(), eps = eps_solver, nswp = 50, kickrank = 4, preconditioner = 'c', verbose = False).cpu()
dofs_tt = tntt.solvers.amen_solve(M_tt, rhs_tt, x0 = tntt.ones(rhs_tt.N), eps = eps_solver, nswp = 60, kickrank = 4, preconditioner = 'c', verbose = False)
tme = datetime.datetime.now() - tme
print('Time system solve ',tme,flush=True)
tme_solve = tme.total_seconds()


print('residual TT ',(M_tt@dofs_tt-rhs_tt).norm()/rhs_tt.norm())
print('residual ref TT ',(M_tt@uref_fun.dofs-rhs_tt).norm()/rhs_tt.norm())
print('residual BD ' , (Pbd_tt@dofs_tt-Pbd_tt@uref_fun.dofs).norm()/Pbd_tt.norm())
print('error tens tt', (dofs_tt - uref_fun.dofs).norm()/dofs_tt.norm())




solution = tt_iga.Function(Basis+Basis_param)
solution.dofs = dofs_tt
err_L2 = solution.L2error(uref, geometry_map = geom, level=100) #L2_error(uref, dofs_tt, Basis+Basis_param, [Xk,Yk,Zk],level=100)

err_Linf = err_L2

print('Computed L2 ',err_Linf)

xyz = geom([tn.linspace(0,1,100),tn.linspace(0,1,100), tn.tensor([0.5]), tn.tensor([1.0])])
u = solution([tn.linspace(0,1,100),tn.linspace(0,1,100), tn.tensor([0.5]), tn.tensor([1.0])])

ref = uref(np.concatenate((xyz[0].numpy().flatten()[:,None],xyz[1].numpy().flatten()[:,None],xyz[2].numpy().flatten()[:,None]),-1)).reshape([100,100])

plt.figure()
plt.contourf(xyz[0].full().squeeze(),xyz[1].full().squeeze(),ref, levels = 100)
plt.colorbar()
plt.figure()
plt.contourf(xyz[0].full().squeeze(),xyz[1].full().squeeze(),u.full().squeeze(), levels = 100)
plt.colorbar()