# Attempt to un-AC the AC-ROM code
# Mike's code
from dolfin import *
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import sys
import numpy as np
from mshr import *
from numpy import linalg as LA
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy import sparse, io

t_0 = 0.0
t_f = 10.0
dt = .02
t_num = int((t_f-t_0)/dt)
TOL = 1.0e-20
nu = 1./100.
t = t_0


N = 70 #Mesh size (across length)

bigcirc = Circle(Point(0.0,0,0),1.0)
littlecirc = Circle(Point(0.5,0.0),0.1,100)
domain = bigcirc - littlecirc

mesh = generate_mesh(domain, N)

measure = assemble(1.0*dx(mesh))
#Element Creation
V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh,MixedElement([V,Q])) 
X_test = VectorFunctionSpace(mesh,"CG",2)
Q_test = FunctionSpace(mesh,"CG",1)
#Set up trial and test functions for all parts
(u,p) = TrialFunctions(W)
(v,q) = TestFunctions(W)

(uf, pf) = TrialFunctions(W) 
(vf, qf) = TestFunctions(W)


#Time and Velocity Setup (BE)
tnPlus1_solutions = Function(W)                                                                    
(unPlus1,pnPlus1) = tnPlus1_solutions.split(True)                                                 

tn_solutions = Function(W)                                                                         
(un,pn) = tn_solutions.split(True)    

tnMin1_solutions = Function(W)
(unM1,pnM1)=tnMin1_solutions.split(True)


# Weak Formulation
def a_1(u,v):                                                                                    
    return inner(nabla_grad(u),nabla_grad(v))
def b(u,v,w):                                                                                      
    return .5*(inner(dot(u,nabla_grad(v)),w)-inner(dot(u,nabla_grad(w)),v))                         
def convect(u,v,w):
	return dot(dot(u, nabla_grad(v)), w) 
def c(p,v):
	return inner(p,div(v))

#Boundary Conditions
def boundary(x, on_boundary):
    return on_boundary

noslip = Constant((0.0, 0.0))

class OriginPoint(SubDomain):
    def inside(self, x, on_boundary):
        tol = .001
        return (near(x[0], 0.0)  and  near(x[1], 1.0))

originpoint = OriginPoint()

bcu = DirichletBC(W.sub(0),noslip,boundary)
bcp = DirichletBC(W.sub(1), 0.0,originpoint, 'pointwise')
bcs = [bcu,bcp]

#Forcing, initial conditions

ufile = File("BDF2_ME/resultsV/velocity.pvd")

fx = "-4*x[1]*(1-pow(x[0],2)-pow(x[1],2))"
fy = "4*x[0]*(1-pow(x[0],2)-pow(x[1],2))"
f = Expression((fx,fy),degree = 2)

w = Function(W)

u_init = Expression(('0','0'),degree = 2) 
unM1.assign(interpolate(u_init,W.sub(0).collapse()))
LNSE = inner(unM1,vf)*dx
NSE = (inner(uf,vf) + dt*(inner(grad(uf)*unM1,vf) + nu*inner(grad(uf),grad(vf)) - div(vf)*pf) + qf*div(uf) )*dx
solve(NSE == LNSE, w, bcs)

#Set up NSE
# LHS
a =  (1.0/(2.0*dt))*3.0*inner(u,v)*dx + nu*a_1(u,v)*dx - c(p,v)*dx + c(q,u)*dx + b(2.0*un - unM1,u,v)*dx 
#RHS
F = (1.0/(2.0*dt))*4.0*inner(un,v)*dx  - (1.0/(2.0*dt))*inner(unM1,v)*dx + inner(f,v)*dx  


####Set up POD basis snapshot collection#########################################
tot_rom = 100
nRom = np.zeros(tot_rom,dtype=np.int)
nRomstr = []
for i in range(0,tot_rom):
    nRom[i] = i + 1
    nRomstr.append(str(nRom[i]))

vdof = X_test.dim() 
pdof = Q_test.dim()

step_start = 4./dt #Timestep number where we start collecting snapshots
t_snap_num = 6./dt #total number timesteps during which we take snapshots
nsnsh = 150
snapRatio = int(t_snap_num/nsnsh) # how often we take a snapshot and store it in our correlation matrix

print(nsnsh)
RomIndex = 0
test_num = t_num
#pdb.set_trace()

H_v = np.zeros((vdof,int(nsnsh)))
H_p = np.zeros((pdof,int(nsnsh)))
##################


# Test Lift and Drag??
drag_arr = np.zeros((t_num))
lift_arr = np.zeros((t_num))

Vh = FunctionSpace(mesh,V)
vdExp = Expression(("0","0"),degree=2)
vlExp =Expression(("0","0"),degree=2)
vd = interpolate(vdExp,W.sub(0).collapse())
vl = interpolate(vlExp,W.sub(0).collapse())
class circle_boundary(SubDomain):
	def inside(self,x,on_boundary):
		return on_boundary and ((x[0]-.5)**2 + (x[1])**2 < 0.01 + 3*DOLFIN_EPS)
circle = DirichletBC(Vh,(0.,-1.),circle_boundary())
circle.apply(vd.vector())
circle = DirichletBC(Vh,(1.,0.),circle_boundary())
circle.apply(vl.vector())
def smooth_bridge(t):
    if(t>1+1e-14):
        return 1.0
    elif(abs(1-t)>1e-14):
        return np.exp(-np.exp(-1./(1-t)**2)/t**2)
    else:
        return 1.0# circle.apply(vl.vector())



####Begin Time Evolution########################################################
for jj in range(0,t_num):
	t = t + dt
#	f = f*min(t,1)
	f = Expression(("mint*(-4*x[1]*(1-pow(x[0],2)-pow(x[1],2)))",\
					"mint*(4*x[0]*(1-pow(x[0],2)-pow(x[1],2)))"),degree = 4, mint = smooth_bridge(t))
	print('Numerical Time Level: t = '+ str(t))

	# a = (1./dt)*dot(u,v)*dx+b(un,u,v)*dx+nu*a_1(u,v)*dx-p*div(v)*dx+div(u)*q*dx    \
	# 	+div(u)*div(v)*dx    # Left-hand-side
	# b_rhs = (1./dt)*dot(un,v)*dx+inner(f,v)*dx  
	# 	a = (1./dt)*inner(u,v)*dx+b(un,u,v)*dx+nu*a_1(u,v)*dx-inner(p,div(v))*dx+inner(div(u),q)*dx        # Left-hand-side
	# b_rhs = (1./dt)*inner(un,v)*dx+inner(f,v)*dx  
	#Matrix Assembly
	A = assemble(a)
	B = assemble(F)

	#Application of boundary condition
	[bc.apply(A,B) for bc in bcs]
	solve(A,w.vector(),B)
	(unPlus1,pnPlus1) = w.split(True)
	pnPlus1.vector()[:]= pnPlus1.vector().get_local() - assemble(pnPlus1/measure*dx)*np.ones_like(pnPlus1.vector().get_local())

	# print("Pressure mean " + str(assemble(pnPlus1*dx)))

	unM1.assign(un)
	pnM1.assign(pn)

	un.assign(unPlus1)
	pn.assign(pnPlus1)

	#Print stuff
	# norm_u = np.sqrt(assemble(unPlus1**2*dx))
	# norm_p = np.sqrt(assemble(pnPlus1**2*dx))
	# print(norm_u)
	# print(norm_p)
	# if norm_p > 100:
	# 	print(norm_p)
	# 	print("uggggg")
	# 	exit()

	drag_arr[jj-1] = assemble(nu*a_1(unPlus1,vd)*dx + convect(unPlus1,unPlus1,vd)*dx - inner(pnPlus1,div(vd))*dx)
	lift_arr[jj-1] = assemble(nu*a_1(unPlus1,vl)*dx + convect(unPlus1,unPlus1,vl)*dx - inner(pnPlus1,div(vl))*dx)

	if jj%snapRatio==0:
		ufile << un

	###Collect initial condition for reduced basis
	if(jj == step_start - 1):
		#pdb.set_trace()
		filename_init_v = './POD_vecs/velocity_init.txt'
		filename_init_p = './POD_vecs/pressure_init.txt'
		u_init_hold = un.vector().get_local()
		p_init_hold = pn.vector().get_local()
		np.savetxt(filename_init_v,u_init_hold)
		np.savetxt(filename_init_p,p_init_hold)
	if(jj == step_start):
		#pdb.set_trace()
		filename_init_v = './POD_vecs/velocity2_init.txt'
		filename_init_p = './POD_vecs/pressure2_init.txt'
		u_init_hold = un.vector().get_local()
		p_init_hold = pn.vector().get_local()
		np.savetxt(filename_init_v,u_init_hold)
		np.savetxt(filename_init_p,p_init_hold)

	#Collect snapshots
	if(jj >= step_start):
		if jj % snapRatio == 0:
			#pdb.set_trace()
			H_v[0:vdof:1,RomIndex] = un.vector().get_local()
			H_p[0:pdof:1,RomIndex] = pn.vector().get_local()
			RomIndex = RomIndex + 1

np.savetxt('lift_offline',lift_arr)
np.savetxt('drag_offline', drag_arr)  


time = np.zeros((t_num))
t = 0.0
for jj in range(0,t_num):
    t = t + dt
    time[jj] = t

plt.figure(1)
plt.plot(time,drag_arr,"r", label=r"drag",linewidth =.5 )
plt.xlabel("t")
plt.ylabel("drag")

plt.savefig("drag offline")

plt.figure(2)
plt.plot(time,lift_arr,"k", label=r"lift", linewidth =.5)
plt.xlabel("t")
plt.ylabel("lift")

plt.savefig("lift offline")




print("Begin POD construction")
u2 = TrialFunction ( X_test )
v2 = TestFunction ( X_test )
p2 = TrialFunction ( Q_test )
q2 = TestFunction ( Q_test )
H_v = 1.0/np.sqrt(nsnsh) * H_v
H_p = 1.0/np.sqrt(nsnsh) * H_p
H_v = 1.0/np.sqrt(nsnsh) * H_v
H_p = 1.0/np.sqrt(nsnsh) * H_p

#np.savetxt('POD_snaps.txt',Q)
MassMat_v = assemble(inner(u2,v2)*dx)
# MassMat_p = assemble(inner(p2,q2)*dx)
M_v = MassMat_v.array()
# M_p = MassMat_p.array()
sM_v = sparse.csr_matrix(M_v)
# sM_p = sparse.csr_matrix(M_p)
Ht_v = np.transpose(H_v)
# Ht_p = np.transpose(H_p)
C_temp_v = sM_v.dot(H_v)
C_v = Ht_v.dot(C_temp_v)
# C_temp_p = sM_p.dot(H_p)
# C_p= Ht_p.dot(C_temp_p)

##

j = 100 #max pod vecs
vals_v, vecs_v = sparse.linalg.eigsh(C_v, k=j)
eigvals_v = np.zeros((j,1))
eigvecs_v = np.zeros((vdof,j))
for i in range(0,j):
    #print nRom
	eigvals_v[i] = vals_v[j-i-1]
	temp_v =  vecs_v[0:vdof:1,j-i-1]
	fin_v = H_v.dot(temp_v)
	fin_v = fin_v/np.sqrt(eigvals_v[i])
	eigvecs_v[0:vdof:1,i] = fin_v[0:vdof:1]
print("The eigenvalues associated with our velocity POD matrix are")
print(eigvals_v)
filename_v = './POD_vecs/velocityMat.txt'
filename2_v = './POD_vecs/POD_vals_velocity.txt'
np.savetxt(filename_v,eigvecs_v);
np.savetxt(filename2_v,eigvals_v);


#### #Begin loop here for velocity POD
# for j in range(0,len(nRom)):
# 	vals_v, vecs_v = sparse.linalg.eigsh(C_v, k=nRom[j])
# 	eigvals_v = np.zeros((nRom[j],1))
# 	eigvecs_v = np.zeros((vdof,nRom[j]))
# 	#fin = np.zeros((ndof,1))
# 	for i in range(0,nRom[j]):
# 	    #print nRom
# 		eigvals_v[i] = vals_v[nRom[j]-i-1]
# 		temp_v =  vecs_v[0:vdof:1,nRom[j]-i-1]
# 		fin_v = H_v.dot(temp_v)
# 		fin_v = fin_v/np.sqrt(eigvals_v[i])
# 		eigvecs_v[0:vdof:1,i] = fin_v[0:vdof:1]
# 	print("The eigenvalues associated with our velocity POD matrix are")
# 	print(eigvals_v)
# 	filename_v = './POD_vecs/velocity_'+ nRomstr[j] + '.txt'
# 	filename2_v = './POD_vecs/POD_vals_velocity.txt'
# 	np.savetxt(filename_v,eigvecs_v);
# np.savetxt(filename2_v,eigvals_v);



# ##Begin loop here for pressure POD
# for j in range(0,len(nRom)):
# 	vals_p, vecs_p = sparse.linalg.eigsh(C_p, k=nRom[j])
# 	eigvals_p = np.zeros((nRom[j],1))
# 	eigvecs_p = np.zeros((pdof,nRom[j]))
# 	#fin = np.zeros((ndof,1))
# 	for i in range(0,nRom[j]):
# 	    #print nRom
# 		eigvals_p[i] = vals_p[nRom[j]-i-1]
# 		temp_p =  vecs_p[0:pdof:1,nRom[j]-i-1]
# 		fin_p = H_p.dot(temp_p)
# 		fin_p = fin_p/np.sqrt(eigvals_p[i])
# 		eigvecs_p[0:pdof:1,i] = fin_p[0:pdof:1]
# 	print("The eigenvalues associated with our pressure POD matrix are")
# 	print(eigvals_p)
# 	filename_p = './POD_vecs/pressure_'+ nRomstr[j] + '.txt'
# 	filename2_p = './POD_vecs/POD_vals_pressure.txt'
# 	np.savetxt(filename_p,eigvecs_p);
# np.savetxt(filename2_p,eigvals_p);






