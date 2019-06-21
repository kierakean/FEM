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
import scipy.linalg

#Declare initial setup parameters and stuff
t_init = 4.0
t_final = 10.0
dt = .001
t_num = int((t_final-t_init)/dt)
nu = 1./100. #Modified to match OfflineBE
t = t_init



#Make the domain
circle_outx = 0.0
circle_outy = 0.0
circle_outr = 1.0
circle_inx = 0.5
circle_iny = 0.0
circle_inr = 0.1

N= 65

domain = Circle(Point(circle_outx,circle_outy),circle_outr) - Circle(Point(circle_inx,circle_iny),circle_inr,100)
mesh = generate_mesh ( domain, N )

plot(mesh)

n = FacetNormal(mesh)

#TH Element Creation
V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh,MixedElement([V,Q]))    
X = VectorFunctionSpace(mesh,"CG",2)
Q_test = FunctionSpace(mesh,"CG",1)
#Set up trial and test functions for all parts
(u,p) = TrialFunctions(W)
(v,q) = TestFunctions(W)

# #Soln dummy vector
w = Function(W)

#Time and Velocity Setup (BE)
tnPlus1_solutions = Function(W)                                                                    
(unPlus1,pnPlus1) = tnPlus1_solutions.split(True)                                                 

tn_solutions = Function(W)                                                                         
(un,pn) = tn_solutions.split(True)    


# Weak Formulation
def contract(u,v):
	return inner(nabla_grad(u),nabla_grad(v))

def b(u,v,w):
	return 0.5*(inner(dot(u,nabla_grad(v)),w)-inner(dot(u,nabla_grad(w)),v))

def a_1(u,v):
	return inner(nabla_grad(u),nabla_grad(v))

#Boundary Conditions
def u0_boundary(x, on_boundary):
    return on_boundary

noslip = Constant((0.0, 0.0))

bcu = DirichletBC(W.sub(0),noslip,u0_boundary)

# Loading POD Basis Functions
frameNum = 20 #(per second)
frameRat = int(1/(frameNum*dt))

val_POD_v = 15
filename1 = './POD_vecs/velocityMat.txt' #filename containing the velocity pod basis vectors
POD_mat_v = np.loadtxt(filename1)
dofV,throwaway = np.shape(POD_mat_v)

POD_mat_v=POD_mat_v[:,:val_POD_v]
print(np.shape(POD_mat_v))

dofV,throwaway = np.shape(POD_mat_v)
nRomV = val_POD_v
print("The total number of velocity basis functions:" + str(nRomV))
velocity_paraview_file = File("para_plotting/velocity_solution_div_discBE.pvd")

POD_funcsV = [Function(X) for i in range(0,nRomV)]
for i in range(0,nRomV):
    POD_funcsV[i].vector()[:] = np.array(np.real(POD_mat_v[:,i])) #Copy POD vectors into Fenics Functions

#Define Functions For initial conditions
u0 = Function(X)
#Load in initial conditions
initial_conV = np.loadtxt('./POD_vecs/velocity_init.txt') #Load in intitial condition in this example we chose not to start from T = 0
#Copy over initial conditions
u0.vector()[:] = np.array((initial_conV[:]))

#RHS Function
f = Expression(("-1*4*x[1]*(1 - x[0]*x[0] - x[1]*x[1])","4*x[0]*(1 - x[0]*x[0] - x[1]*x[1])"),pi=np.pi, degree = 4)

#Lift and drag eventually
vdExp = Expression(("0","0"),degree=2)
vlExp =Expression(("0","0"),degree=2)
vd = interpolate(vdExp,X)
vl = interpolate(vlExp,X)
class circle_boundary(SubDomain):
	def inside(self,x,on_boundary):
		return on_boundary and ((x[0]-.5)**2 + (x[1])**2 < 0.01 + 3*DOLFIN_EPS)
circle = DirichletBC(X,(0.,-1.),circle_boundary())
circle.apply(vd.vector())
circle = DirichletBC(X,(1.,0.),circle_boundary())
circle.apply(vl.vector())

#Declarations
H_V = np.zeros((nRomV,nRomV))
T = np.zeros((nRomV,nRomV,nRomV))
D = np.zeros((nRomV,nRomV))
bv = np.zeros((nRomV,1))
bv1 = np.zeros((nRomV,1))
bv2 = np.zeros((nRomV,1))
xv = np.zeros((nRomV,1)) #un in ROM basis


#Assemble the offline systems
for i in range(0,nRomV):
	bv1[i]= assemble(inner(POD_funcsV[i],f)*dx)
	for j in range(0,nRomV):
		H_V[i,j] = assemble((1./dt)*inner(POD_funcsV[i],POD_funcsV[j])*dx + nu*a_1(POD_funcsV[i],POD_funcsV[j])*dx) #LHS Matrix
		D[i,j] = assemble((1./dt)*inner(POD_funcsV[i],POD_funcsV[j])*dx) #RHS
		for k in range(0,nRomV):
			T[k,i,j]= T[k,i,j]+assemble(b(POD_funcsV[k],POD_funcsV[j],POD_funcsV[i])*dx) #LHS matrix (nonlinear)

#Time Evolution
for jj in range(0,t_num):
	t = t + dt
	print('Numerical Time Level: t = '+ str(t))
	if jj < 2:
		Av = np.zeros((nRomV,nRomV))
		C = np.zeros((nRomV,nRomV))
		for i in range(0,nRomV):
			for j in range(0,nRomV):
				C[i,j] = -1*assemble(b(u0,POD_funcsV[i],POD_funcsV[j])*dx) #The nonlinear thing
		for i in range(0,nRomV):
			for j in range(0,nRomV):
				 Av[i,j] = H_V[i,j] + C[i,j] #The entire matrix for LHS
		luV,piV = scipy.linalg.lu_factor(Av) #LU decomposition

		for i in range(0,nRomV):
			bv[i] = assemble(inner(POD_funcsV[i],f)*dx + (1.0/dt)*inner(u0,POD_funcsV[i])*dx )

	else:
		bv = bv1.copy()
		#Velocity Matrix
		Av = np.zeros((nRomV,nRomV))
		C = np.zeros((nRomV,nRomV))
		for i in range(0,nRomV):
			for j in range(0,nRomV):
				bv[i] = bv[i] + xv[j]*D[i,j]
				for k in range(0,nRomV):
					C[i,j] = C[i,j]+xv[k]*T[k,i,j]#The nonlinear thing

	 	#The entire matrix for LHS
		Av= H_V+C
		luV,piV = scipy.linalg.lu_factor(Av) #LU decomposition
	xv = scipy.linalg.lu_solve((luV,piV),bv)
	
	if jj%frameRat==0: # only save un in full basis when we want to make a calculation/print
		sol_V = POD_mat_v.dot(xv) #
		un.vector()[:] = np.array(np.real(sol_V[:,0])) #
		velocity_paraview_file << (un,t)



















