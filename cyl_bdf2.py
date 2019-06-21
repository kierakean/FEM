# Attempt at flow around a cylander 2d benchmark problem
# Time discritization with BDF2
# From demos, and code from Victor, Mike, and Michael
from fenics import *
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from ufl import *
from dolfin import *
from mshr import *
import sympy as sp 
import matplotlib.pyplot as plt 

#Parameters, given by paper/chosen based on given codes

nu = 1/1000

#Use time 0-8 for benchmark
T_0 = 0.0
T_f = 8.0 
t = T_0
dt = .001
TOL = 1.0e-20
t_num = int((T_f-T_0)/dt)


#Setup inner products -----------------

def a_1(u,v):
	return inner(nabla_grad(u),nabla_grad(v))
def b(u,v,w):
	return .5*(inner(dot(u,nabla_grad(v)),w)-inner(dot(u,nabla_grad(w)),v))  
def convect(u,v,w):
	return dot(dot(u, nabla_grad(v)), w) 
def c(p,v):
	return inner(p,div(v))

# Set up the domain and geometry
N = 64 #Mesh size (across length)

channel = Rectangle(Point(0.0),Point(2.2,0.41))
cylindar = Circle(Point(0.2,0.2),0.05)
domain = channel - cylindar

mesh = Mesh('cylinder.finer.mesh.xml.gz')


V = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh,MixedElement([V,Q]))    

(u,p) = TrialFunctions(W)
(v,q) = TestFunctions(W)

(uf, pf) = TrialFunctions(W) #For bdf2 part I think
(vf, qf) = TestFunctions(W)


w = Function(W)

# Boundary Things --------------------
def boundary(x,on_boundary):
	return on_boundary

walls = 'near(x[1],0) || near(x[1],0.41)'
inflow = 'near(x[0],0)'
outflow = 'near(x[0],2.2)'
cyl = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

class OriginPoint(SubDomain):                                                                    
    def inside(self,x,on_boundary):
        return x[0] < TOL and x[1] < TOL
originpoint = OriginPoint()

# Actual boundary conditions (inflow velocity and no slip

InflowExp = Expression(('pow(0.41,-2)*sin(pi*t/8.0)*6*x[1]*(0.41-x[1])','0'),t=t,degree = 2)

bcw = DirichletBC(W.sub(0),Constant((0,0)), walls) #bc for top/bottom walls
bcc = DirichletBC(W.sub(0),Constant((0,0)), cyl) #bc for cylindar
bci = DirichletBC(W.sub(0),InflowExp,inflow)
bco = DirichletBC(W.sub(0),InflowExp,outflow)
                                                   
bcp = DirichletBC(W.sub(1),0.0,originpoint,'pointwise')  
bcs = [bcw,bcc,bci,bco,bcp]

#Create Storage Files
ufile = File("BDF2Results/resultsV/velocity.pvd")
pfile = File("BDF2Results/resultsP/pressure.pvd")


#Time and Velocity Setup 
tnPlus1_solutions = Function(W)                                                                    
(unPlus1,pnPlus1) = tnPlus1_solutions.split(True)                                                 

tn_solutions = Function(W)                                                                         
(un,pn) = tn_solutions.split(True)    

tnMin1_solutions = Function(W)
(unM1,pnM1)=tnMin1_solutions.split(True)


# Get u0 and u1 so BDF2 works

f = Expression(("0.0","0.0"),degree = 4)
u_init = Expression(('0','0'),degree = 2) 
unM1.assign(interpolate(u_init,W.sub(0).collapse()))
LNSE = inner(unM1,vf)*dx
NSE = (inner(uf,vf) + dt*(inner(grad(uf)*unM1,vf) + nu*inner(grad(uf),grad(vf)) - div(vf)*pf) + qf*div(uf) )*dx
solve(NSE == LNSE, w, bcs)


#Set up problem for NSE (From Mike and Michael)
# LHS
a =  (1.0/(2.0*dt))*3.0*inner(u,v)*dx + nu*a_1(u,v)*dx - c(p,v)*dx + c(q,u)*dx + b(2.0*un - unM1,u,v)*dx 

#RHS
F = (1.0/(2.0*dt))*4.0*inner(un,v)*dx  - (1.0/(2.0*dt))*inner(unM1,v)*dx


# Blank arrays to store quantites of interest
drag_arr = np.zeros((t_num))
lift_arr = np.zeros((t_num))
press_arr = np.zeros((t_num))
drag_arr_line = np.zeros((t_num))
lift_arr_line = np.zeros((t_num))


#Test functions for volumetric calculations
Vh = FunctionSpace(mesh,V)
vl = Function(W.sub(0).collapse())
vd = Function(W.sub(0).collapse())

vdExp = Expression(("0","0"),degree=2)
vlExp =Expression(("0","0"),degree=2)
vd = interpolate(vdExp,W.sub(0).collapse())
vl = interpolate(vlExp,W.sub(0).collapse())

class circle_boundary(SubDomain):
	def inside(self,x,on_boundary):
		return on_boundary and \
				((x[0]-.2)**2 + (x[1]-.2)**2 < 0.0025 + 3*DOLFIN_EPS)

circle = DirichletBC(Vh,(1,0),circle_boundary())
circle.apply(vd.vector())
circle = DirichletBC(Vh,(0,1),circle_boundary())
circle.apply(vl.vector())

#Stuff for line integral calculations
# n = FacetNormal(mesh)

# boundaries = FacetFunction("size_t", mesh)
# boundaries.set_all(0)

# I = Identity(unM1.geometric_dimension())

# circle_boundary().mark(boundaries,1)
# ds = Measure("ds")[boundaries]

#Time Stepping
count = 0
while t <= T_f + TOL:
	print('Numerical time level: t = ',t)
	
	InflowExp.t=t
	
	A = assemble(a)
	B = assemble(F)

	[bc.apply(A,B) for bc in bcs]
	solve(A,w.vector(),B)
	(unPlus1,pnPlus1) = w.split(True)

	press_arr[count-1]= pn(0.15,0.2)-pn(0.25,0.2)
	
	#Volume Calculation
	drag_arr[count-1] = -20*assemble(nu*a_1(unPlus1,vd)*dx + convect(unPlus1,unPlus1,vd)*dx - c(pnPlus1,vd)*dx)
	lift_arr[count-1] = -20*assemble(nu*a_1(unPlus1,vl)*dx + convect(unPlus1,unPlus1,vl)*dx - c(pnPlus1,vl)*dx)
	
	#Line integral Calculation
	# D = 0.5*(grad(un)+grad(un).T)
	# Tensor = -pn*I + 2*nu*D
	# force = Tensor*nu
	# drag_arr_line[count-1]=-assemble(force[1]*ds(1))
	# lift_arr = assemble(force[0]*ds(1))

	unM1.assign(un)
	pnM1.assign(pn)

	un.assign(unPlus1)
	pn.assign(pnPlus1)

	t += dt
	count += 1
	if count%30==0:
		ufile << un
		pfile << pn   



#Output and plot pressure difference, drag, and lift

time = np.zeros((t_num))
t = 0.0
for jj in range(0,t_num):
    t = t + dt
    time[jj] = t


print("The Maximum Drag Is:" + str(np.max(drag_arr)))
print("The Maximum Lift Is" + str(np.max(lift_arr)))

np.savetxt('pressure_difference',press_arr)
np.savetxt('lift_arr',lift_arr)
np.savetxt('drag_arr',drag_arr)

plt.figure(1)
plt.plot(time,drag_arr,"r", label=r"drag",linewidth =.5 )
plt.xlabel("t")
plt.ylabel("drag")

plt.savefig("drag")

plt.figure(2)
plt.plot(time,lift_arr,"k", label=r"lift", linewidth =.5)
plt.xlabel("t")
plt.ylabel("lift")

plt.savefig("lift")

#plt.show()    
plt.interactive(True)




























