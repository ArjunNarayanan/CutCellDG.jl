using PyPlot
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("../../test/useful_routines.jl")
include("analytical-solver.jl")
include("transformation-elasticity-solver.jl")
PS = PlaneStrainSolver
TES = TransformationElasticitySolver

function lame_lambda(k, m)
    return k - 2m / 3
end

function angular_position(points)
    cpoints = points[1, :] + im * points[2, :]
    return rad2deg.(angle.(cpoints))
end

polyorder = 3
nelmts = 17
penaltyfactor = 1e3

width = 1.0              # mm
K1, K2 = 247.0, 192.0    # GPa
mu1, mu2 = 126.0, 87.0   # GPa

lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)
stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)

rho1 = 3.93e3           # Kg/m^3
rho2 = 3.68e3           # Kg/m^3
V01 = 1.0 / rho1
V02 = 1.0 / rho2

ΔG0Jmol = -14351.0  # J/mol
molarmass = 0.147
ΔG0 = ΔG0Jmol / molarmass

theta0 = -0.067
transfstress =
    CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)

meshwidth = [width, width]
numqp = required_quadrature_order(polyorder) + 2

dx = width / nelmts
penalty = penaltyfactor / dx * 0.5 * (lambda1 + mu1 + lambda2 + mu2)

basis = TensorProductBasis(2, polyorder)
interfacecenter = [0.5, 0.5]
interfaceradius = 0.3
outerradius = 1.5
analyticalsolution = PS.CylindricalSolver(
    interfaceradius,
    outerradius,
    interfacecenter,
    lambda1,
    mu1,
    lambda2,
    mu2,
    theta0,
)

# mesh, cellquads, facequads, interfacequads, levelset =
#     TES.construct_mesh_and_quadratures(
#         meshwidth,
#         nelmts,
#         basis,
#         x -> -circle_distance_function(x, interfacecenter, interfaceradius),
#         numqp,
#     )

cgmesh = CutCellDG.CGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basis)
dgmesh = CutCellDG.DGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basis)
levelset = CutCellDG.LevelSet(
    x -> -circle_distance_function(x, interfacecenter, interfaceradius),
    cgmesh,
    basis,
)


cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
refseedpoints, refseedcellids =
    CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
spatialseedpoints =
    CutCellDG.map_to_spatial(refseedpoints, refseedcellids, cutmesh)
dx = minimum(CutCellDG.element_size(dgmesh))
penalty = penaltyfactor / dx * 0.5 * (lambda1 + mu1 + lambda2 + mu2)


# analyticalsolution = PS.CylindricalSolver(
#     interfaceradius,
#     outerradius,
#     interfacecenter,
#     lambda1,
#     mu1,
#     lambda2,
#     mu2,
#     theta0,
# )
#
# potentialdifference = TES.potential_difference_at_nodal_coordinates(
#     cutmesh,
#     basis,
#     levelset,
#     refseedpoints,
#     refseedcellids,
#     spatialseedpoints,
#     stiffness,
#     theta0,
#     V01,
#     V02,
#     ΔG0,
#     numqp,
#     penalty,
#     analyticalsolution,
#     1e-12,
#     4.5,
# )
#
# newcoeffs = TES.step_levelset(
#     levelset,
#     potentialdifference,
#     refseedpoints,
#     refseedcellids,
#     spatialseedpoints,
#     1e-12,
#     4.5,
# )



coeffs0 = CutCellDG.coefficients(levelset)
coeffs1 = TES.step_interface(
    dgmesh,
    basis,
    levelset,
    stiffness,
    theta0,
    V01,
    V02,
    ΔG0,
    numqp,
    interfacecenter,
    outerradius,
)
CutCellDG.update_coefficients!(levelset, coeffs1)
cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
refseedpoints, refseedcellids =
    CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
spatialseedpoints =
    CutCellDG.map_to_spatial(refseedpoints, refseedcellids, cutmesh)
nodalcoordinates =
    CutCellDG.nodal_coordinates(CutCellDG.background_mesh(levelset))
signeddistance = CutCellDG.distance_to_zero_levelset(
    nodalcoordinates,
    refseedpoints,
    spatialseedpoints,
    refseedcellids,
    levelset,
    0.1,
    4.5,
)




querypoint = nodalcoordinates[:, 124]
xguess = refseedpoints[:, 7]
guesscellid = refseedcellids[7]
cellmap = CutCellDG.cell_map(dgmesh, 7)
CutCellDG.load_coefficients!(levelset, guesscellid)

func = CutCellDG.interpolater(levelset)
grad(x) = vec(gradient(func, x))
hess(x) = CutCellDG.hessian_matrix(func, x)

xc = CutCellDG.saye_newton_iterate(
    xguess,
    querypoint,
    func,
    grad,
    hess,
    cellmap,
    1e-12,
    4.5,
)


using NearestNeighbors
tree = KDTree(spatialseedpoints)
seedidx, seeddists = nn(tree, nodalcoordinates)
idx = 124
sidx = seedidx[idx]
xguess = refseedpoints[:, sidx]
xquery = nodalcoordinates[:, idx]
guesscellid = refseedcellids[sidx]

cellmap = CutCellDG.cell_map(CutCellDG.background_mesh(levelset), guesscellid)
cellmap = CutCellDG.cell_map(dgmesh, guesscellid)
CutCellDG.load_coefficients!(levelset, guesscellid)
func = CutCellDG.interpolater(levelset)
grad(x) = vec(gradient(func, x))
hess(x) = CutCellDG.hessian_matrix(func, x)

jac = CutCellDG.jacobian(dgmesh)
xq = xquery
x0 = copy(xguess)
gp = grad(x0)
l0 = gp' * ((xq - cellmap(x0)) .* jac) / (gp' * gp)

function run_saye_iterations(x0, xq, func, grad, hess, cellmap, jac, numiter)
    xiter = zeros(2, numiter + 1)
    xiter[:, 1] = x0
    gp = grad(x0)
    l0 = gp' * ((xq - cellmap(x0)) .* jac) / (gp' * gp)
    for iter = 1:numiter
        xnext, lnext = CutCellDG.step_saye_newton_iterate(
            xiter[:, iter],
            l0,
            xq,
            func,
            grad,
            hess,
            cellmap,
            jac,
            2,
            1e5eps(),
            2.0,
        )
        xiter[:, iter+1] = xnext
        l0 = lnext
    end
    return xiter
end

xiter = run_saye_iterations(x0, xq, func, grad, hess, cellmap, jac, 20)

using Plots
xrange = -1:1e-2:1
Plots.contour(xrange, xrange, (x, y) -> levelset([x, y]), levels = [0.0])
Plots.scatter!(xiter[1,:],xiter[2,:])

refcp = CutCellDG.saye_newton_iterate(
    xguess,
    xquery,
    func,
    grad,
    hess,
    cellmap,
    1e-12,
    4.5,
    maxiter=100
)



refclosestpoints, refclosestcellids, refgradients =
    CutCellDG.closest_reference_points_on_levelset(
        nodalcoordinates,
        refseedpoints,
        spatialseedpoints,
        refseedcellids,
        levelset,
        1e-12,
        4.5,
        100
    )




# coeffs2 = TES.step_interface(
#     dgmesh,
#     basis,
#     levelset,
#     stiffness,
#     theta0,
#     V01,
#     V02,
#     ΔG0,
#     numqp,
#     interfacecenter,
#     outerradius,
# )
#
# oldcoeffs = CutCellDG.coefficients(levelset)
#
# fig, ax = PyPlot.subplots()
# ax.tricontour(nodalcoordinates[1, :], nodalcoordinates[2, :], oldcoeffs, [0.0])
# ax.tricontour(nodalcoordinates[1, :], nodalcoordinates[2, :], newcoeffs, [0.0])
# ax.set_aspect("equal")
# ax.grid()
# fig
