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
import pdb
import os
from offline_supremizer_stabalization import *


rrr = 0
for line in open("input_info_C_in_C.txt"):
    li=line.strip()
    if not li.startswith("#"):
        if rrr == 0:
            t_0 = float(line.rstrip())
        if rrr == 1:
            t_final = float(line.rstrip())
        if rrr == 2:
            nu = float(line.rstrip())
        if rrr == 3:
            N = int(line.rstrip())
        if rrr == 4:
        	N_outer = int(line.rstrip())
        if rrr == 5:
        	N_inner = int(line.rstrip())
        if rrr == 6:
        	t_init = float(line.rstrip())
        rrr = rrr + 1
#Selection of number of basis functions
val_POD_v = 20
val_POD_p = 5
nRomP = val_POD_p
nRomV = val_POD_v

#Time setup
dt = .00025
t_num = int((t_final-t_init)/dt)
#nu = 1./100. 
t = t_init

#Make the domain
circle_outx = 0.0
circle_outy = 0.0
circle_outr = 1.0
circle_inx = 0.5
circle_iny = 0.0
circle_inr = 0.1


# N_outer = 400
# N_inner = 100
# N =85

domain = Circle(Point(circle_outx,circle_outy),circle_outr,N_outer) - Circle(Point(circle_inx,circle_iny),circle_inr,N_inner)
mesh = generate_mesh ( domain, N )

n = FacetNormal(mesh)

V = VectorElement("Lagrange", mesh.ufl_cell(), 2)

#TH Element Creation

X = VectorFunctionSpace(mesh,"CG",2)
Q = FunctionSpace(mesh,"CG",1)

u = TrialFunction(X)
p = TrialFunction(Q) #supremizer stablization

v = TestFunction(X)
q = TestFunction(Q)


un = Function(X)
pn_s = Function(Q)
pn_p = Function(Q)
p0_s = Function(Q)
p0_p = Function(Q)


# Weak Formulations
def a_1(u,v):                                                                                    
    return inner(nabla_grad(u),nabla_grad(v))
def contract(u,v):
	return inner(nabla_grad(u),nabla_grad(v))
def b(u,v,w):                                                                                      
    return .5*(inner(dot(u,nabla_grad(v)),w)-inner(dot(u,nabla_grad(w)),v))                         
def convect(u,v,w):
	return dot(dot(u, nabla_grad(v)), w) 
def c(p,v):
	return inner(p,div(v))
# For Pressure Poisson 
def cross_2(xx,yy):
    return xx[0]*yy[1]-xx[1]*yy[0] 
f = Expression(("-1*4*x[1]*(1 - x[0]*x[0] - x[1]*x[1])","4*x[0]*(1 - x[0]*x[0] - x[1]*x[1])"),pi=np.pi, degree = 4)


#Define Functions For initial conditions
u0 = Function(X)
#Load in initial conditions
initial_conV = np.loadtxt('./POD_vecs/velocity_init.txt') #Load in intitial condition in this example we chose not to start from T = 0
#Copy over initial conditions
u0.vector()[:] = np.array((initial_conV[:]))

#Pressure initial condition
p0 = Function(Q)
#Load in initial conditions
initial_conP = np.loadtxt('./POD_vecs/pressure_init.txt') #Load in intitial condition in this example we chose not to start from T = 0
#Copy over initial conditions
p0.vector()[:] = np.array((initial_conP[:]))

#Lift and drag
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

drag_arr_sup = np.zeros((t_num))
lift_arr_sup = np.zeros((t_num))

drag_arr_pp = np.zeros((t_num))
lift_arr_pp = np.zeros((t_num))

drag_diff_arr_s = np.zeros((t_num))
lift_diff_arr_s = np.zeros((t_num))

drag_diff_arr_pp = np.zeros((t_num))
lift_diff_arr_pp = np.zeros((t_num))

press_norm_pp = np.zeros((t_num))
press_norm_s = np.zeros((t_num))

### Drag and lift Calculations
drag_temp = assemble(nu*a_1(u0,vd)*dx + convect(u0,u0,vd)*dx - c(p0,vd)*dx)
lift_temp = assemble(nu*a_1(u0,vl)*dx + convect(u0,u0,vl)*dx - c(p0,vl)*dx)
# drag_arr_sup[0] = drag_temp
# lift_arr_sup[0] = lift_temp
# drag_arr[jj] = assemble(nu*a_1(unPlus1,vd)*dx + convect(unPlus1,unPlus1,vd)*dx - c(pnPlus1,vd)*dx)
# lift_arr[jj] = assemble(nu*a_1(unPlus1,vl)*dx + convect(unPlus1,unPlus1,vl)*dx - c(pnPlus1,vl)*dx)


# drag_offline = np.loadtxt('array_folder/drag_offline_Late.txt')
# lift_offline = np.loadtxt('array_folder/lift_offline_Late.txt')
drag_offline = np.loadtxt('array_folder/drag_offline.txt')
lift_offline = np.loadtxt('array_folder/lift_offline.txt')
index_start = int(t_init/dt) - 1
print('The difference between the initial drag and lift:' + str(drag_offline[index_start] - drag_temp))

drag_arr_off = np.zeros((t_num))
lift_arr_off = np.zeros((t_num))


for i in range(0,t_num):
	drag_arr_off[i] = drag_offline[i+index_start]
	lift_arr_off[i] = lift_offline[i+index_start]

#pdb.set_trace()


# Loading POD Basis Functions Velocity 
filename1 = './POD_vecs/velocityMatBE.txt' #filename containing the velocity pod basis vectors
POD_mat_v = np.loadtxt(filename1)
dofV,throwaway = np.shape(POD_mat_v)

POD_mat_v=POD_mat_v[:,:val_POD_v]
dofV,throwaway = np.shape(POD_mat_v)
nRomV = val_POD_v
print("The total number of velocity basis functions:" + str(nRomV))
POD_funcsV = [Function(X) for i in range(0,nRomV)]
for i in range(0,nRomV):
    POD_funcsV[i].vector()[:] = np.array(np.real(POD_mat_v[:,i])) #Copy POD vectors into Fenics Functions

xv_arr = np.loadtxt('xv_arr.txt')
print(np.shape(xv_arr))


v_lam_hold = Function(X)
p_lam_hold = Function(Q)
fname_singular_values_V = './POD_vecs/POD_vals_velocityBE.txt'
fname_singular_values_P = './POD_vecs/POD_vals_pressureBE.txt'
v_sing_vals = np.loadtxt(fname_singular_values_V)
p_sing_vals = np.loadtxt(fname_singular_values_P)

lam_val_v = 0
lam_val_p = 0 

for i in range(val_POD_v,throwaway):
	v_lam_hold.vector()[:] = np.array(np.real(POD_mat_v[:,i]))
	grad_phi_val = assemble(inner(nabla_grad(v_lam_hold),nabla_grad(v_lam_hold))*dx)
	lam_val_v = lam_val_v + grad_phi_val*v_sing_vals[i]
	lam_val_p = lam_val_p + p_sing_vals[i]



#pdb.set_trace()




#POD basis functions pressure
#POD basis functions pressure
filename2 = './POD_vecs/pressureMatBE.txt' #filename containing the velocity pod basis vectors
POD_mat_p_star = np.loadtxt(filename2)
dofP,throwaway = np.shape(POD_mat_p_star)

POD_funcsP = [Function(Q) for i in range(0,nRomV)]
for i in range(0,nRomV):
    POD_funcsP[i].vector()[:] = np.array(np.real(POD_mat_p_star[:,i])) #Copy POD vectors into Fenics Functions


##Measure the projection error###
coeff_holdV = np.zeros(nRomV)
coeff_holdP = np.zeros(nRomP)
p_diff = Function(Q)
POD_p_proj = Function(Q)
v_diff = Function(X)
POD_v_proj = Function(X)
for i in range(0,nRomV):
	coeff_holdV[i] = assemble(inner(u0,POD_funcsV[i])*dx)
	POD_v_proj.vector()[:] = POD_v_proj.vector()[:] + coeff_holdV[i]*POD_funcsV[i].vector()[:]
for i in range(0,nRomP):
	coeff_holdP[i] = assemble(inner(p0,POD_funcsP[i])*dx)
	POD_p_proj.vector()[:] = POD_p_proj.vector()[:] + coeff_holdP[i]*POD_funcsP[i].vector()[:]

p_diff.vector()[:] = p0.vector()[:] - POD_p_proj.vector()[:]
v_diff.vector()[:] = u0.vector()[:] - POD_v_proj.vector()[:]

err_vel = np.sqrt(assemble(dot(v_diff,v_diff)*dx))
err_press = np.sqrt(assemble(dot(p_diff,p_diff)*dx))
print("the error in the velocity is:" + str(err_vel))
print("the error in the pressure is:" + str(err_press))

drag_temp = assemble(nu*a_1(POD_v_proj,vd)*dx + convect(POD_v_proj,POD_v_proj,vd)*dx - c(POD_p_proj,vd)*dx)
lift_temp = assemble(nu*a_1(POD_v_proj,vl)*dx + convect(POD_v_proj,POD_v_proj,vl)*dx - c(POD_p_proj,vl)*dx)
drag_arr_sup[0] = drag_temp
lift_arr_sup[0] = lift_temp

drag_arr_pp[0] = drag_temp
lift_arr_pp[0] = lift_temp



#RHS Function


#Space Preallocation
#Velocity

#Pressure
H_PP_star =  np.zeros((nRomP,nRomP))
K_p = np.zeros((nRomV,nRomP))
T2_p = np.zeros((nRomV,nRomV,nRomP))
bp_p_f = np.zeros((nRomP,1))

#Pressure Poisson
H_PP_star =  np.zeros((nRomP,nRomP))
K_p = np.zeros((nRomV,nRomP))
T2_p = np.zeros((nRomV,nRomV,nRomP))
bp_p_f = np.zeros((nRomP,1))

#Lift Drag
PL1 = np.zeros((nRomP,1))
PD1 = np.zeros((nRomP,1))
PL2 = np.zeros((nRomV,1))
PD2 = np.zeros((nRomV,1))
PL3 = np.zeros((nRomV,nRomV))
PD3 = np.zeros((nRomV,nRomV))




print("Now Doing the Assembly")

#Assemble the offline systems
for i in range(0,nRomV):
	for j in range(0,nRomP):
		K_p[i,j] = nu*assemble(dot(curl(POD_funcsV[i]),cross_2(n,grad(POD_funcsP[j])))*ds) 
	for j in range(0,nRomV):
		for k in range(0,nRomP):
			T2_p[i,j,k] = assemble(convect(POD_funcsV[i],POD_funcsV[j],grad(POD_funcsP[k]))*dx) #RHS vector for pressure (nonlinear part)

for i in range(0,nRomP):
	bp_p_f[i]= assemble(inner(grad(POD_funcsP[i]),f)*dx)
	for j in range(0,nRomP):
		H_PP_star[i,j] = assemble(a_1(POD_funcsP[i],POD_funcsP[j])*dx) #RHS for pressure matrix (never changes)

#Assemble for lift and drag
for i in range(0,nRomP):
	PL1[i] = assemble(inner(POD_funcsP[i],div(vl))*dx)  #preassembly lift part 1
	PD1[i] = assemble(inner(POD_funcsP[i],div(vd))*dx) 
for j in range(0,nRomV):
	PL2[j] = nu*assemble(a_1(POD_funcsV[j],vl)*dx)
	PD2[j] = nu*assemble(a_1(POD_funcsV[j],vd)*dx)
	for k in range(0,nRomV):
		PL3[j,k] = assemble(convect(POD_funcsV[j],POD_funcsV[k],vl)*dx)
		PD3[j,k] = assemble(convect(POD_funcsV[j],POD_funcsV[k],vd)*dx)


#pdb.set_trace()




nRomP_arr = [3,6,9,12,15,18,21,24,27,30,40,50]
nRomP_arr = [1,2,3,5]
for s in range(0,12):
	t = t_init
	
	nRomP = nRomP_arr[s]
	POD_mat_p=POD_mat_p_star[:,:nRomP]

	# print("The total number of pressure basis functions:" + str(nRomP))

	coeff_holdP = np.zeros(nRomP)
	p_diff = Function(Q)
	POD_p_proj = Function(Q)

	for i in range(0,nRomP):
		coeff_holdP[i] = assemble(inner(p0,POD_funcsP[i])*dx)
		POD_p_proj.vector()[:] = POD_p_proj.vector()[:] + coeff_holdP[i]*POD_funcsP[i].vector()[:]

	p_diff.vector()[:] = p0.vector()[:] - POD_p_proj.vector()[:]

	err_press = np.sqrt(assemble(dot(p_diff,p_diff)*dx))


	# print("the projection error in the pressure is:" + str(err_press))
	print("getting supremizer functions")

	Sup_mat,inf_sup_const = offline_supremizer_stabalization(int(N),int(N_outer),int(N_inner),nRomP,dofP,POD_mat_p)
	# print("The inf-sup constant associated with our supremizer basis:" + str(inf_sup_const))

	Sup_funcs = [Function(X) for i in range(0,nRomP)]
	for i in range(0,nRomP):
	    Sup_funcs[i].vector()[:] = np.array(np.real(Sup_mat[:,i])) #Copy POD vectors into Fenics Functions




	#Preallocation
	bp_p =np.zeros((nRomP,1))
	H_PP = np.zeros((nRomP,nRomP))

	#Pressure
	H_PS = np.zeros((nRomP,nRomP))
	A_s = np.zeros((nRomV,nRomP))
	D2_s = np.zeros((nRomV,nRomP))
	T2_s = np.zeros((nRomV,nRomV,nRomP))
	xp_s = np.zeros((nRomP,1))
	bp_s_f = np.zeros((nRomP,1))
	
	print("assembling")
	for i in range(0,nRomV):
		for j in range(0,nRomP):
			
			D2_s[i,j] = assemble(inner(POD_funcsV[i],Sup_funcs[j])*dx) #Supremizer time discretization term
			for k in range(0,nRomP):
				T2_s[i,j,k] = assemble(b(POD_funcsV[i],POD_funcsV[j],Sup_funcs[k])*dx) #Sup Stabilizer nonlinear
	for i in range(0,nRomP):
		bp_s_f[i] = -1*assemble(inner(f,Sup_funcs[i])*dx) #rhs force
		for j in range(0,nRomP):
			H_PS[i,j] = assemble(inner(POD_funcsP[i],div(Sup_funcs[j]))*dx) #unchanging LHS matrix for pressure recovery
			H_PP[i,j] = H_PP_star[i,j]

	luP_s,piP_s = scipy.linalg.lu_factor(H_PS)
	luP_p,piP_p = scipy.linalg.lu_factor(H_PP)

	print("timestepping")
	for jj in range(0,t_num):
		t = t + dt
		xvM1 = xv_arr[:,[jj]] 
		xv =  xv_arr[:,[jj+1]] 

		bp_s = np.zeros((nRomP,1))

		for i in range(0,nRomP):
			bp_s[i]= bp_s_f[i]
			bp_p[i] = bp_p_f[i]
			for j in range(0,nRomV):
				bp_s[i] = bp_s[i] + (1./dt)*(xv[j]-xvM1[j])*D2_s[j,i]
				bp_p[i] = bp_p[i] + xv[j]*K_p[j,i]
				for k in range(0,nRomV):
					bp_s[i] = bp_s[i] + xvM1[j]*xv[k]*T2_s[j,k,i] 
					bp_p[i] = bp_p[i]- xvM1[j]*xv[k]*T2_p[j,k,i]
		
		xp_s = scipy.linalg.lu_solve((luP_s,piP_s),bp_s)

		xp_p = scipy.linalg.lu_solve((luP_p,piP_p),bp_p)

		drag_s = 0.
		lift_s = 0.

		drag_p = 0.
		lift_p = 0.

		for j in range(0,nRomV):
			drag_s = drag_s + xv[j]*PD2[j]
			lift_s = lift_s + xv[j]*PL2[j]

			drag_p = drag_p + xv[j]*PD2[j]
			lift_p = lift_p + xv[j]*PL2[j]
			for k in range(0,nRomV):
				drag_s = drag_s + xv[j]*xv[k]*PD3[j,k]
				lift_s = lift_s + xv[j]*xv[k]*PL3[j,k]

				drag_p = drag_p + xv[j]*xv[k]*PD3[j,k]
				lift_p = lift_p + xv[j]*xv[k]*PL3[j,k]

		for i in range(0,nRomP):
			drag_s = drag_s - xp_s[i]*PD1[i] 
			lift_s = lift_s - xp_s[i]*PL1[i]

			drag_p = drag_p - xp_p[i]*PD1[i] 
			lift_p = lift_p - xp_p[i]*PL1[i]
		drag_arr_sup[jj+1] = drag_s
		lift_arr_sup[jj+1] = lift_s

		drag_arr_pp[jj+1] = drag_p
		lift_arr_pp[jj+1] = lift_p

		# print('The difference between the drag (sup stab):' + str(drag_offline[index_start+jj+1] - drag_s))
		# print('The difference between the lift (sup stab):' + str(lift_offline[index_start+jj+1] - lift_s))

		# print('The difference between the drag (pressure poisson):' + str(drag_offline[index_start+jj+1] - drag_p))
		# print('The difference between the lift (pressure poisson):' + str(lift_offline[index_start+jj+1] - lift_p))
	np.savetxt('array_folder/lift_online_sup_stab'+str(nRomV)+"_"+str(nRomP) + ".txt",lift_arr_sup)
	np.savetxt('array_folder/drag_online_sup_stab'+str(nRomV)+"_"+str(nRomP) + ".txt",drag_arr_sup) 
	np.savetxt('array_folder/lift_online_press_poiss'+str(nRomV)+"_"+str(nRomP) + ".txt",lift_arr_pp)
	np.savetxt('array_folder/drag_online_press_poiss'+str(nRomV)+"_"+str(nRomP) + ".txt",drag_arr_pp)   
	np.savetxt('array_folder/press_arr_sup_stab'+str(nRomV)+"_"+str(nRomP) + '.txt',press_norm_s)
	np.savetxt('array_folder/press_arr_press_poiss'+str(nRomV)+"_"+str(nRomP) + '.txt',press_norm_pp)



	# plt.plot(time2,drag_arr_sup,"r", label=r"drag",linewidth =.5 )
	# plt.xlabel("t")
	# plt.ylabel("drag")
	# plt.savefig("pic_folder/drag_online_sup_stab"+str(nRomV)+"_"+str(nRomP))
	# plt.close()

	# plt.plot(time2,lift_arr_sup,"k", label=r"lift", linewidth =.5)
	# plt.xlabel("t")
	# plt.ylabel("lift")
	# plt.savefig("pic_folder/lift_online_sup_stab"+str(nRomV)+"_"+str(nRomP))
	# plt.close()



	# plt.plot(time2,drag_arr_pp,"r", label=r"drag",linewidth =.5 )
	# plt.xlabel("t")
	# plt.ylabel("drag")
	# plt.savefig("pic_folder/drag_online_press_poiss"+str(nRomV)+"_"+str(nRomP))
	# plt.close()

	# plt.plot(time2,lift_arr_pp,"k", label=r"lift", linewidth =.5)
	# plt.xlabel("t")
	# plt.ylabel("lift")
	# plt.savefig("pic_folder/lift_online_press_poiss"+str(nRomV)+"_"+str(nRomP))
	# plt.close()



	plt.plot(time2,drag_arr_pp,"r", label=r"drag pp",linewidth =.5 )
	plt.plot(time2,drag_arr_sup,"g", label=r"drag ss",linewidth =.5 )
	plt.plot(time2,drag_arr_off,"b", label=r"drag offline",linewidth =.5 )
	plt.xlabel("t")
	plt.xlabel("t")
	plt.ylabel("drag")
	plt.savefig("pic_folder/drag_online_comps"+str(nRomV)+"_"+str(nRomP))
	plt.close()



	plt.plot(time2,lift_arr_pp,"r", label=r"lift pp",linewidth =.5 )
	plt.plot(time2,lift_arr_sup,"g", label=r"lift ss",linewidth =.5 )
	plt.plot(time2,lift_arr_off,"b", label=r"lift offline",linewidth =.5 )
	plt.xlabel("t")
	plt.xlabel("t")
	plt.ylabel("drag")
	plt.savefig("pic_folder/lift_online_comps"+str(nRomV)+"_"+str(nRomP))
	plt.close()


	POD_mat_v=POD_mat_v[:,:val_POD_v]
	time3 = time2[:800]
	drag3_pp= drag_arr_pp[:800]
	drag3_s =drag_arr_sup[:800]
	drag3_off =  drag_arr_off[:800]



	time3 = time3[1:]
	drag3_pp = drag3_pp[1:]
	drag3_s = drag3_s[1:]
	drag3_off = drag3_off[1:]


	plt.plot(time3,drag3_pp,"r", label=r"drag pp",linewidth =.5 )
	plt.plot(time3,drag3_s,"g", label=r"drag ss",linewidth =.5 )
	plt.plot(time3,drag3_off,"b", label=r"drag offline",linewidth =.5 )
	plt.xlabel("t")
	plt.xlabel("t")
	plt.ylabel("drag")
	plt.savefig("pic_folder/drag_online_comps_small"+str(nRomV)+"_"+str(nRomP))
	plt.close()














