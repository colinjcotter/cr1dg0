from firedrake import *

n = 100
mesh = PeriodicUnitSquareMesh(n, n)

V = VectorFunctionSpace(mesh, "CR", 1)
Q = FunctionSpace(mesh, "DG", 0)

x, y = SpatialCoordinate(mesh)

p_exact = exp(sin(2*pi*x)*sin(2*pi*y))
u_exact = as_vector([2*pi*cos(2*pi*x),2*pi*cos(2*pi*y)])*p_exact
div_u_exact = -4*pi**2*(-sin(2*pi*x) + cos(2*pi*x)**2
                        - sin(2*pi*y) + cos(2*pi*x)**2)*p_exact

V1 = FunctionSpace(mesh, "BDM", 1)
Q1 = FunctionSpace(mesh, "DG", 0)

W = V * Q
W1 = V1 * Q1

v, q = TestFunctions(W)
u, p = TrialFunctions(W)

a = inner(u,v)*dx + p*div(v)*dx + q*div(u)*dx + q*p*dx
L = (div_u_exact+p_exact)*q*dx

wCR = Function(W, name="CR")
CRprob = LinearVariationalProblem(a, L, wCR)
CRsolver = LinearVariationalSolver(CRprob)

v, q = TestFunctions(W1)
u, p = TrialFunctions(W1)

a = inner(u,v)*dx + p*div(v)*dx + q*div(u)*dx + q*p*dx
L = (f+p_exact)*q*dx

wBDM = Function(W1, name="BDM")
BDMprob = LinearVariationalProblem(a, L, wBDM)
BDMsolver = LinearVariationalSolver(BDMprob)

CRsolver.solve()
BDMsolver.solve()

u, p = wCR.split()
u1, p1 = wBDM.split()

File('CRtest.pvd').write(u, p, u1, p1)

