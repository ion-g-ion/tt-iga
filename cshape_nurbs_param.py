import torchtt as tntt
import numpy as np
import tt_iga
import matplotlib.pyplot as plt
import torch as tn
import datetime

tn.set_default_dtype(tn.float64)

r = 1.0
R = 2.0
delta = 0.5
nl = 12
control_points = tt_iga.geometry.ParameterDependentControlPoints([2,3])
weights = tt_iga.geometry.ParameterDependentWeights([2,3])

basis1 = tt_iga.bspline.BSplineBasis(np.linspace(0,1,2),1)
basis2 = tt_iga.bspline.BSplineBasis(np.linspace(0,1,2),2)

basis_params = [tt_iga.lagrange.LagrangeLeg(nl)]*2

weights[:,0] = 1.0
weights[:,-1] = 1.0
weights[:,1] = 1/np.sqrt(2)

control_points[:,0,0] = [-R,0]
control_points[:,1,0] = [-r,0]
control_points[:,0,2] = [0,R]
control_points[:,1,2] = [0,r]
control_points[:,1,1] = [-r,r]
control_points[:,0,1] = [lambda params: -R+delta*params[0],lambda params: R+delta*params[1]]

geom = tt_iga.PatchNURBS.interpolate_parameter_dependent(control_points,weights,[basis1, basis2],basis_params)



y1,y2 = tn.linspace(0,1,100), tn.linspace(0,1,100)
xs = geom([y1, y2, tn.tensor([-1]), tn.tensor([-1])])
plt.figure()
plt.scatter(xs[0].numpy().flatten(),xs[1].numpy().flatten(),s=1)
xs = geom([y1, y2, tn.tensor([1]), tn.tensor([1])])
plt.scatter(xs[0].numpy().flatten(),xs[1].numpy().flatten(),s=1)
xs = geom([y1, y2, tn.tensor([0]), tn.tensor([0])])
plt.scatter(xs[0].numpy().flatten(),xs[1].numpy().flatten(),s=1)


 

basis_solution = [tt_iga.bspline.BSplineBasis(np.concatenate((np.linspace(0,0.4,32), np.linspace(0.4,0.6,16),np.linspace(0.6,1,32))),2)]
basis_solution.append(tt_iga.bspline.BSplineBasis(np.concatenate((np.linspace(0,0.15,20),np.linspace(0.15,0.3,20), np.linspace(0.3,0.5,20),np.linspace(0.5,1,20))),2))
Mass_tt = geom.mass_interp(basis_solution)
Stiff_tt = geom.stiffness_interp(basis_solution)

Jref = lambda y: y[...,0]*0.0 + 1.0

rhs_tt = geom.rhs_interp(basis_solution,Jref)

P1 = tn.eye(Mass_tt.N[0])
P2 = tn.eye(Mass_tt.N[0])
P2[-1,-1] = 0
P2[0,0] = 0
P1[0,0] = 0
P1[-1,-1] = 0
Pin_tt = tntt.rank1TT([P1,P2]) ** tntt.eye([nl]*2)
Pbd_tt = (tntt.eye(Mass_tt.N) - Pin_tt) ** tntt.eye([nl]*2)

M_tt = (Pin_tt@Stiff_tt+Pbd_tt).round(1e-12)
rhs_tt = (Pin_tt @ rhs_tt + 0).round(1e-12)

print('System matrix... ',flush=True)


print('Rank Mtt ',M_tt.R)
print('Rank rhstt ',rhs_tt.R)

tme = datetime.datetime.now() 
# dofs_tt = tntt.solvers.amen_solve(M_tt.cuda(), rhs_tt.cuda(), x0 = tntt.ones(rhs_tt.N).cuda(), eps = eps_solver, nswp = 50, kickrank = 4, preconditioner = 'c', verbose = False).cpu()
dofs_tt = tntt.solvers.amen_solve(M_tt, rhs_tt, x0 = tntt.ones(rhs_tt.N), eps = 1e-8, nswp = 60, kickrank = 4, preconditioner = 'c', verbose = True)
tme = datetime.datetime.now() - tme
print('Time system solve ',tme,flush=True)



plt.figure()
y1, y2 = np.linspace(0,1,201), np.linspace(0.,1,201)
X1,X2 = geom([y1, y2, tn.tensor([1]), tn.tensor([1])])
u = dofs_tt.mprod([tn.tensor(basis_solution[0](y1).T),tn.tensor(basis_solution[1](y2).T),tn.tensor(basis_params[0](tn.tensor([1])).T),tn.tensor(basis_params[1](tn.tensor([1])).T)],[0,1,2,3])
plt.contourf(X1.numpy().squeeze(), X2.numpy().squeeze() ,u.numpy().squeeze() ,levels=32)
plt.colorbar()