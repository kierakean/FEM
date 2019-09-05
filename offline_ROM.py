#ROM code, more organized, Offline Section

#Victor, Mike, and Kiera's code

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
import pdb


rrr = 0
for line in open("input_info_C_in_C.txt"):
    li=line.strip()
    if not li.startswith("#"):
        if rrr == 0:
            t_0 = float(line.rstrip())
        if rrr == 1:
            t_f = float(line.rstrip())
        if rrr == 2:
            nu = float(line.rstrip())
        if rrr == 3:
            N = int(line.rstrip())
        if rrr == 4:
        	N_outer = int(line.rstrip())
        if rrr == 5:
        	N_inner = int(line.rstrip())
        if rrr == 6:
        	t_start = float(line.rstrip())
        if rrr ==7:
        	dt =  float(line.rstrip())
        rrr = rrr + 1

#t_0 = 0.0 #Start time
#t_f = 6.0 #End time
dt = .0005 #Timestep
t_num = int((t_f-t_0)/dt) #total number of steps
#nu = 1./200. #Viscosity 
t = t_0 #initialize time

#Generalized Offset Cylinders
circle_outx = 0.0
circle_outy = 0.0
circle_outr = 1.0
circle_inx = 0.5
circle_iny = 0.0
circle_inr = 0.1

 #Create Mesh and Domains
# N = 45
# N_outer = 30
# N_inner = 30


#t_start = 4. #Time where we start taking snapshots
step_start = int(t_start/dt)-1 #Timestep number where we start collecting snapshots
t_snap_poss = int((t_f-t_start)/dt)+1 #total number of time steps during which we may take snapshots
snapRatio = 1 # how often we take a snapshot and store it in our correlation matrix
nsnsh = int((t_snap_poss/snapRatio)) # total number of snapshots

domain = Circle(Point(circle_outx,circle_outy),circle_outr,N_outer) - Circle(Point(circle_inx,circle_iny),circle_inr,N_inner)
mesh = generate_mesh ( domain, N )

measure = assemble(1.0*dx(mesh)) #size of the domain (for normalizing pressure)


#TH Element Creation
V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh,MixedElement([V,Q]))    
X_test = VectorFunctionSpace(mesh,"CG",2)
Q_test = FunctionSpace(mesh,"CG",1)
vdof = X_test.dim() 
pdof = Q_test.dim()

print("The number of velocity DOFs is:" + str(vdof))
print("The number of pressure DOFs is:" + str(pdof))
print("The total number of timesteps is:" + str(t_num))


#Set up trial and test functions for all parts
(u,p) = TrialFunctions(W)
(v,q) = TestFunctions(W)

#Soln dummy vector
w = Function(W)

#Create solution locatioms
tnPlus1_solutions = Function(W)                                                                    
(unPlus1,pnPlus1) = tnPlus1_solutions.split(True)                                                 

tn_solutions = Function(W)                                                                         
(un,pn) = tn_solutions.split(True)   


#File Storage for Paraview Plotting
#frameNum = 1./dt #(paraview snaps per second)
frameRat = 10
# frameRat = 100
#frameRat = int(1/(frameNum*dt))
#pdb.set_trace()
#pdb.set_trace()

ufile = File("BE_ME/resultsV/velocity.pvd")
pfile = File("BE_ME/resultsP/pressure.pvd")

# Weak Formulations
def a_1(u,v):                                                                                    
    return inner(nabla_grad(u),nabla_grad(v))
def b(u,v,w):                                                                                      
    return .5*(inner(dot(u,nabla_grad(v)),w)-inner(dot(u,nabla_grad(w)),v))                         
def convect(u,v,w):
	return dot(dot(u, nabla_grad(v)), w) 
def c(p,v):
	return inner(p,div(v))

#Boundary Conditions
def u0_boundary(x, on_boundary):
    return on_boundary

noslip = Constant((0.0, 0.0))


mesh_points = mesh.coordinates()
class OriginPoint(SubDomain):
    def inside(self, x, on_boundary):
        tol = .001
        return (near(x[0], mesh_points[0,0])  and  near(x[1], mesh_points[0,1]))
# class OriginPoint(SubDomain):
#     def inside(self, x, on_boundary):
#         tol = .001
#         return (near(x[0], 1.0)  and  near(x[1], 0.0))

originpoint = OriginPoint()

bcu = DirichletBC(W.sub(0),noslip,u0_boundary)
bcp = DirichletBC(W.sub(1), 0.0, originpoint, 'pointwise')
bcs = [bcu,bcp]

#Initial Conditions (Start from rest)
u_init = Expression(('0','0'),degree = 2) 
un.assign(interpolate(u_init,W.sub(0).collapse()))




#Smooth bridge (to allow force to increase slowly)
def smooth_bridge(t):
    if(t>1+1e-14):
        return 1.0
    elif(abs(1-t)>1e-14):
        return np.exp(-np.exp(-1./(1-t)**2)/t**2)
    else:
        return 1.0

f = Expression(("mint*(-4*x[1]*(1-pow(x[0],2)-pow(x[1],2)))",\
					"mint*(4*x[0]*(1-pow(x[0],2)-pow(x[1],2)))"),degree = 4, mint= 0.0)

#Forcing
# fx = "-4*x[1]*(1-pow(x[0],2)-pow(x[1],2))"
# fy = "4*x[0]*(1-pow(x[0],2)-pow(x[1],2))"
# f = Expression((fx,fy),degree = 4)

mint_val = smooth_bridge(t)
#f.mint = t_f

#NSE
a = (1./dt)*inner(u,v)*dx+b(un,u,v)*dx+nu*a_1(u,v)*dx-c(p,v)*dx+c(q,u)*dx  # Left-hand-side
F = (1./dt)*inner(un,v)*dx+inner(f,v)*dx  #Right hand side

## Test Lift and Drag
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


####Set up POD basis snapshot collection#########################################





RomIndex = 0 #Index for snapshot matrix

H_v = np.zeros((vdof,int(nsnsh)))
H_p = np.zeros((pdof,int(nsnsh)))





#Timestepping
for jj in range(0,t_num):
	t = t + dt
	mint_val = smooth_bridge(t)
	f.mint = mint_val
	#pdb.set_trace() 

	print('Numerical Time Level: t = '+ str(t))

	#Matrix Assembly
	A = assemble(a)
	B = assemble(F)

	#Application of boundary condition
	[bc.apply(A,B) for bc in bcs]
	#Solve
	solve(A,w.vector(),B)
	#Solution
	(unPlus1,pnPlus1) = w.split(True)
	#Shift so mean is zero
	pnPlus1.vector()[:]= pnPlus1.vector().get_local() - assemble(pnPlus1/measure*dx)*np.ones_like(pnPlus1.vector().get_local())

	#step forward in time
	un.assign(unPlus1)
	pn.assign(pnPlus1)
	energy = 0.5*dot(un, un)*dx
	E = assemble(energy)
	print("The energy is:" + str(E))


	### Drag Calculations
	drag_arr[jj] = assemble(nu*a_1(unPlus1,vd)*dx + convect(unPlus1,unPlus1,vd)*dx - c(pnPlus1,vd)*dx)
	lift_arr[jj] = assemble(nu*a_1(unPlus1,vl)*dx + convect(unPlus1,unPlus1,vl)*dx - c(pnPlus1,vl)*dx)

	if abs(drag_arr[jj]) > 10:
		print('help')
		ufile << un
		# pfile << pn
		exit()


	#Take a paraview shot
	#pdb.set_trace()
	if jj%frameRat==0:
		ufile << (un,t)
		pfile << (pn,t)

	#Collect initial condition
	if(jj == step_start):
		print("I collected an initial condition at t =" + str(t))
		#pdb.set_trace()
		filename_init_v = './POD_vecs/velocity_init.txt'
		filename_init_p = './POD_vecs/pressure_init.txt'
		u_init_hold = un.vector().get_local()
		p_init_hold = pn.vector().get_local()
		np.savetxt(filename_init_v,u_init_hold)
		np.savetxt(filename_init_p,p_init_hold)

	#Begin taking snapshots 
	if(jj >= step_start): #Take snapshots for every time step from 12 to 16 
		if jj % snapRatio == 0:
			#Add snapshots to matrix
			H_v[0:vdof:1,RomIndex] = un.vector().get_local()
			H_p[0:pdof:1,RomIndex] = pn.vector().get_local()
			RomIndex = RomIndex + 1





###Plotting Lift and Drag
#Save array
np.savetxt('V_snaps.txt',H_v)
np.savetxt('P_snaps.txt',H_p)

np.savetxt('Array_Folder/lift_offline.txt',lift_arr)
np.savetxt('Array_Folder/drag_offline.txt', drag_arr)  

#entire time
time1 = np.zeros((t_num))
t = 0.0
for jj in range(0,t_num):
    t = t + dt
    time1[jj] = t

plt.figure(1)
plt.plot(time1,drag_arr,"r", label=r"drag",linewidth =.5 )
plt.xlabel("t")
plt.ylabel("drag")

plt.savefig("pic_folder/full_drag_offline")

plt.figure(2)
plt.plot(time1,lift_arr,"k", label=r"lift", linewidth =.5)
plt.xlabel("t")
plt.ylabel("lift")

plt.savefig("pic_folder/full_lift_offline")


time2 = np.zeros(t_snap_poss)
#t_num2 = int((t_f-t_start)/dt)
start2 = int(t_start/dt)-1
drag2 = np.zeros(t_snap_poss)
lift2 = np.zeros(t_snap_poss)
t = t_start
pdb.set_trace()
for jj in range(0,t_snap_poss):
	t = t + dt
	time2[jj] = t
	drag2[jj] = drag_arr[start2+jj]
	lift2[jj] = lift_arr[start2+jj]


plt.figure(3)
plt.plot(time2,drag2,"r", label=r"drag",linewidth =.5 )
plt.xlabel("t")
plt.ylabel("drag")

plt.savefig("pic_folder/drag_offline")

plt.figure(4)
plt.plot(time2,lift2,"k", label=r"lift", linewidth =.5)
plt.xlabel("t")
plt.ylabel("lift")

plt.savefig("pic_folder/lift_offline")

# Pod construction
print("Begin POD construction")
u2 = TrialFunction ( X_test )
v2 = TestFunction ( X_test )
p2 = TrialFunction ( Q_test )
q2 = TestFunction ( Q_test )
H_v = 1.0/np.sqrt(nsnsh) * H_v
H_p = 1.0/np.sqrt(nsnsh) * H_p
Ht_v = np.transpose(H_v)
Ht_p = np.transpose(H_p)

#Create sparse matrices to ensure no size limitaitons
parameters['linear_algebra_backend'] = 'Eigen'
temp_form = inner(u2,v2)*dx
A_1 = assemble(temp_form)

temp_form2 = inner(p2,q2)*dx
B_1 = assemble(temp_form2)

row_A,col_A,val_A = as_backend_type(A_1).data()
row_B,col_B,val_B = as_backend_type(B_1).data()
sA = sparse.csr_matrix((val_A,col_A,row_A))
sB = sparse.csr_matrix((val_B,col_B,row_B))

C_temp_v = sA.dot(H_v)
C_v = Ht_v.dot(C_temp_v)

C_temp_p = sB.dot(H_p)
C_p= Ht_p.dot(C_temp_p)

#Velocity Pod
j = 70 #max pod vecs
vals_v, vecs_v = sparse.linalg.eigsh(C_v, k=j)
eigvals_v = np.zeros((j,1))
eigvecs_v = np.zeros((vdof,j))
for i in range(0,j):
    #print nRom
	eigvals_v[i] = vals_v[j-i-1]
	temp_v =  vecs_v[:,j-i-1]
	fin_v = H_v.dot(temp_v)
	fin_v = fin_v/np.sqrt(eigvals_v[i])
	eigvecs_v[:,i] = fin_v[:]
print("The eigenvalues associated with our velocity POD matrix are")
print(eigvals_v)
filename_v = './POD_vecs/velocityMatBE.txt'
filename2_v = './POD_vecs/POD_vals_velocityBE.txt'
np.savetxt(filename_v,eigvecs_v);
np.savetxt(filename2_v,eigvals_v);

#PressurePod
vals_p, vecs_p = sparse.linalg.eigsh(C_p, k=j)
eigvals_p = np.zeros((j,1))
eigvecs_p = np.zeros((pdof,j))
#fin = np.zeros((ndof,1))
for i in range(0,j):
    #print nRom
    eigvals_p[i] = vals_p[j-i-1]
    temp_p =  vecs_p[:,j-i-1]
    fin_p = H_p.dot(temp_p)
    fin_p = fin_p/np.sqrt(eigvals_p[i])
    eigvecs_p[:,i] = fin_p[:]
print("The eigenvalues associated with our pressure POD matrix are")
print(eigvals_p)
filename_p = './POD_vecs/pressureMatBE.txt'
filename2_p = './POD_vecs/POD_vals_pressureBE.txt'
np.savetxt(filename_p,eigvecs_p);
np.savetxt(filename2_p,eigvals_p);






# if t_f > 12:
# 	time3 = np.zeros((int(((t_f-12.)/dt))))
# 	t_num3 = int((t_f-12.)/dt)
# 	start3 = int(12./dt)
# 	drag3 = np.zeros((t_num3))
# 	lift3 = np.zeros((t_num3))
# 	t = 12
# 	for jj in range(0,t_num3):
# 		t = t + dt
# 		time3[jj] = t
# 		drag3[jj] = drag_arr[start3+jj]
# 		lift3[jj] = lift_arr[start3+jj]


# 	plt.figure(5)
# 	plt.plot(time3,drag3,"r", label=r"drag",linewidth =.5 )
# 	plt.xlabel("t")
# 	plt.ylabel("drag")

# 	plt.savefig("pic_folder/drag_offline_end")

# 	plt.figure(6)
# 	plt.plot(time3,lift3,"k", label=r"lift", linewidth =.5)
# 	plt.xlabel("t")
# 	plt.ylabel("lift")


# 	plt.savefig("pic_folder/lift_offline_end")















