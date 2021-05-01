using PyPlot
using Statistics
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("../../../test/useful_routines.jl")
include("transformation-elasticity-solver.jl")
include("analytical-solver.jl")
TES = TransformationElasticitySolver
APS = AnalyticalPlaneSolver


meshwidth = [1.0, 1.0]
polyorder = 3
nelmts = 27
penaltyfactor = 1e4
numqp = required_quadrature_order(polyorder) + 2

K1, K2 = 247.0, 247.0    # GPa
mu1, mu2 = 126.0, 126.0   # GPa

lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)
stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)

rho1 = 3.93e3           # Kg/m^3
rho2 = 3.68e3           # Kg/m^3
V01 = 1.0 / rho1
V02 = 1.0 / rho2

# ΔG0Jmol = -14351.0  # J/mol
ΔG0Jmol = -2e4
molarmass = 0.147
ΔG0 = ΔG0Jmol / molarmass

theta0 = -0.067
transfstress =
    CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)

basis = TensorProductBasis(2, polyorder)
initialposition = 0.51
frequency = 1.5
amplitude = 0.2
CFL = 0.25

function perturbation(x, frequency, amplitude)
    return amplitude * sin.(2 * pi * frequency * x)
end

function plot_zero_levelset(coeffs, nodalcoordinates)
    fig, ax = PyPlot.subplots()
    ax.tricontour(nodalcoordinates[1, :], nodalcoordinates[2, :], coeffs, [0.0])
    ax.grid()
    ax.set_aspect("equal")
    return fig
end

distancefunction(x) =
    plane_distance_function(x, [1.0, 0.0], [initialposition, 0.0]) +
    perturbation(x[2, :], frequency, amplitude)

cgmesh = CutCellDG.CGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basis)
dgmesh = CutCellDG.DGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basis)
levelset = CutCellDG.LevelSet(distancefunction, cgmesh, basis)
coeffs0 = copy(CutCellDG.coefficients(levelset))
paddedmesh = CutCellDG.BoundaryPaddedMesh(cgmesh, 1)
nodalcoordinates = CutCellDG.nodal_coordinates(cgmesh)

nodalcoordinates = CutCellDG.nodal_coordinates(cgmesh)
elementsize = CutCellDG.element_size(cgmesh)
minelmtsize = minimum(elementsize)
maxelmtsize = maximum(elementsize)
penalty = penaltyfactor / minelmtsize * 0.5 * (lambda1 + mu1 + lambda2 + mu2)
tol = minelmtsize^(polyorder + 1)
boundingradius = 1.5 * maxelmtsize

numquerypoints = 100
querypoints = vcat(
    repeat([interfacepos0], numquerypoints)',
    range(0, 1, length = numquerypoints)',
)
ycoords = querypoints[2, :]

cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
refseedpoints, seedcellids = CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
spatialseedpoints =
    CutCellDG.map_to_spatial(refseedpoints, seedcellids, cutmesh)
R0 = mean(spatialseedpoints[1, :])
dev0 = std(spatialseedpoints[1, :])

pd0 = TES.potential_difference_at_query_points(
    querypoints,
    cutmesh,
    basis,
    levelset,
    stiffness,
    theta0,
    numqp,
    penalty,
    spatialseedpoints,
    seedcellids,
    V01,
    V02,
    ΔG0,
    tol,
    boundingradius,
)

shift = mean(pd0)
abspd0 = abs.(pd0)
maxpd0 = maximum(pd0)
minpd0 = minimum(pd0)
scale = maxpd0 - minpd0

interfaceperturbation =
    shift .+
    0.5 * scale / amplitude * perturbation.(ycoords, frequency, amplitude)
fig, ax = PyPlot.subplots()
ax.plot(ycoords, pd0, color = "black", label = L"[\Phi]")
ax.plot(
    ycoords,
    interfaceperturbation,
    "--",
    color = "black",
    label = "interface perturbation",
)
ax.grid()
ax.legend()
ax.set_ylim(minpd0 - 0.1 * scale, maxpd0 + 0.1 * scale)
fig
