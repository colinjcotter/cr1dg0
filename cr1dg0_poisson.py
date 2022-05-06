from firedrake import *

n = 100
mesh = PeriodicUnitSquareMesh(n, n)
Vc = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)

# (1, 0) -> (1, 0)
# (0, 1) -> (1/2, sqrt(3)/2)



# (X) = (1 1/2      )(x)
# (Y) = (0 sqrt(3)/2)(y)

# (x) = (2/sqrt(3))(sqrt(3)/2 -1/2)(X) = (1  -1/sqrt(3))(X)
# (y) =            (0            1)(Y)   (0   1/sqrt(3))(Y)

f = Function(Vc).interpolate(as_vector([x + y/2, sqrt(3)/2*y]))
mesh.coordinates.assign(f)

V = VectorFunctionSpace(mesh, "CR", 1)
Q = FunctionSpace(mesh, "DG", 0)

x0, y0 = SpatialCoordinate(mesh)
Vx = FunctionSpace(mesh, "CG", 1)
x = Function(Vx).interpolate(x0)
y = Function(Vx).interpolate(y0)

p_exact = exp(sin(2*pi*(x - y/sqrt(3)))*sin(2*pi*y/sqrt(3)*2))
u_exact = as_vector([diff(p_exact, x), diff(p_exact, y)])
div_u_exact = diff(u_exact[0], x) + diff(u_exact[1], y)

V1 = FunctionSpace(mesh, "BDM", 1)
Q1 = FunctionSpace(mesh, "DG", 0)

W = V * Q
W1 = V1 * Q1

v, q = TestFunctions(W)
u, p = TrialFunctions(W)

# u - grad(p) = 0, p + div(u) = f

a = inner(u,v)*dx + p*div(v)*dx + q*div(u)*dx + q*p*dx
L = (div_u_exact+p_exact)*q*dx

wCR = Function(W, name="CR")
CRprob = LinearVariationalProblem(a, L, wCR)

params = {'mat_type':'aij',
          'ksp_type':'preonly',
          'pc_type':'lu',
          'pc_factor_mat_solver_type':'mumps'}

CRsolver = LinearVariationalSolver(CRprob, solver_parameters=params)

v, q = TestFunctions(W1)
u, p = TrialFunctions(W1)

a = inner(u,v)*dx + p*div(v)*dx + q*div(u)*dx + q*p*dx
L = (div_u_exact+p_exact)*q*dx

wBDM = Function(W1, name="BDM")
BDMprob = LinearVariationalProblem(a, L, wBDM)
BDMsolver = LinearVariationalSolver(BDMprob, solver_parameters=params)

CRsolver.solve()
BDMsolver.solve()

u, p = wCR.split()
u1, p1 = wBDM.split()

u_proj = Function(V1, name="Projected CR")
ut = TrialFunction(V1)
v = TestFunction(V1)
n = FacetNormal(mesh)
a = inner(v('+'), n('+'))*inner(ut('+'), n('+'))*dS
L = inner(v('+'), n('+'))*inner(avg(u1), n('+'))*dS
projprob = LinearVariationalProblem(a, L, u_proj)
projsolver = LinearVariationalSolver(projprob)
projsolver.solve()

Vcg = FunctionSpace(mesh, "CG", 1)
Vcg2 = VectorFunctionSpace(mesh, "CG", 1)
p_exact_out = Function(Vcg, name="p_exact").interpolate(p_exact)
u_exact_out = Function(Vcg2, name="u_exact").interpolate(u_exact)

Vdg = FunctionSpace(mesh, "DG", 1)
Vdg2 = VectorFunctionSpace(mesh, "DG", 1)
p_cr_error = Function(Vdg, name="p_cr_error").project(p - p_exact_out)
p_bdm_error = Function(Vdg, name="p_bdm_error").project(p1 - p_exact_out)
u_cr_error = Function(Vdg2, name="u_cr_error").project(u - u_exact_out)
u_proj_cr_error = Function(Vdg2,
                           name="u_proj_cr_error").project(u_proj - u_exact_out)
u_bdm_error = Function(Vdg2, name="u_bdm_error").project(u1 - u_exact_out)

File('CRtest.pvd').write(u, p, u1, p1, u_proj, p_exact_out, u_exact_out,
                         p_cr_error, p_bdm_error, u_bdm_error, u_cr_error,
                         u_proj_cr_error)
