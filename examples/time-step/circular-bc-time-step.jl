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
interfacecenter = [0.0, 0.0]
interfaceradius = 0.5
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

mesh, cellquads, facequads, interfacequads, levelset =
    TES.construct_mesh_and_quadratures(
        meshwidth,
        nelmts,
        basis,
        x -> -circle_distance_function(x, interfacecenter, interfaceradius),
        numqp,
    )

nodaldisplacement = TES.nodal_displacement(
    mesh,
    basis,
    cellquads,
    facequads,
    interfacequads,
    stiffness,
    theta0,
    analyticalsolution,
    penalty,
)

querypoints = CutCellDG.nodal_coordinates(CutCellDG.background_mesh(levelset))
cutmesh = CutCellDG.background_mesh(mesh)
refseedpoints, refseedcellids =
    CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
spatialseedpoints =
    CutCellDG.map_to_spatial(refseedpoints, refseedcellids, cutmesh)

potentialdifference =
    (
        ΔG0 .+
        1e9 * TES.potential_difference_at_closest_points(
            querypoints,
            nodaldisplacement,
            basis,
            refseedpoints,
            refseedcellids,
            spatialseedpoints,
            mesh,
            levelset,
            stiffness,
            theta0,
            V01,
            V02,
        )
    ) / abs(ΔG0)

paddedmesh =
    CutCellDG.BoundaryPaddedMesh(CutCellDG.background_mesh(levelset), 1)
tol = 1e-3
boundingradius = 10.0
paddedlevelset = CutCellDG.BoundaryPaddedLevelSet(
    paddedmesh,
    refseedpoints,
    spatialseedpoints,
    refseedcellids,
    levelset,
    tol,
    boundingradius,
)

bgmesh = CutCellDG.background_mesh(levelset)
bgcoords = CutCellDG.bottom_ghost_coordinates(paddedmesh)
xq = bgcoords[:, 2]

using NearestNeighbors
tree = KDTree(spatialseedpoints)
seedidx, seeddists = nn(tree, xq)
guesscellid = refseedcellids[seedidx]
xg = refseedpoints[:, seedidx]
cellmap = CutCellDG.cell_map(bgmesh, guesscellid)
CutCellDG.load_coefficients!(levelset, guesscellid)
poly = CutCellDG.interpolater(levelset)

refcp = CutCellDG.saye_newton_iterate(
    xg,
    xq,
    poly,
    x -> vec(gradient(poly, x)),
    x -> CutCellDG.hessian_matrix(poly, x),
    cellmap,
    1e-5,
    7.0,
)

using Plots
xrange = -1.5:1e-2:1.5
Plots.contour(xrange, xrange, (x, y) -> levelset([x, y]), levels = [0.0])
Plots.scatter!([xg[1]], [xg[2]])

# exactpotentialdifference =
#     (ΔG0 + 1e9*PS.interface_potential_difference(analyticalsolution,V01,V02))/abs(ΔG0)

# pderr =
#     maximum(abs.(potentialdifference .- exactpotentialdifference)) /
#     abs(exactpotentialdifference)
# println("Normalized error in potential difference = $pderr")

# Δylim = 0.1 * abs(exactpotentialdifference)
# fig, ax = PyPlot.subplots()
# numpts = length(potentialdifference)
# ax.scatter(1:numpts, potentialdifference, s = 0.1)
# ax.plot(
#     1:numpts,
#     exactpotentialdifference * ones(numpts),
#     "--",
# )
# ax.set_ylim(exactpotentialdifference - Δylim, exactpotentialdifference + Δylim)
# ax.grid()
# fig
