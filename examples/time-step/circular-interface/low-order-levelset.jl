using PyPlot
using LinearAlgebra
using Statistics
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

function potential_error(pd, exactpd)
    return maximum(abs.(pd .- exactpd)) / abs(exactpd)
end

function exact_normalized_potential_difference(solver, V01, V02, ΔG0)
    return (ΔG0 + 1e9PS.interface_potential_difference(solver, V01, V02)) /
           abs(ΔG0)
end

levelsetorder = 2
elasticityorder = 3
nelmts = 33
penaltyfactor = 1e2

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
numqp = required_quadrature_order(elasticityorder) + 2

levelsetbasis = TensorProductBasis(2, levelsetorder)
elasticitybasis = TensorProductBasis(2, elasticityorder)
interfacecenter = [0.5, 0.5]
interfaceradius = 0.4
outerradius = 1.2

cgmesh =
    CutCellDG.CGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], levelsetbasis)
dgmesh =
    CutCellDG.DGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], elasticitybasis)
levelset = CutCellDG.LevelSet(
    x -> -circle_distance_function(x, interfacecenter, interfaceradius),
    cgmesh,
    levelsetbasis,
)
coeffs0 = CutCellDG.coefficients(levelset)
levelsetcoords = CutCellDG.nodal_coordinates(cgmesh)

dx = minimum(CutCellDG.element_size(dgmesh))
penalty = penaltyfactor / dx * 0.5 * (lambda1 + mu1 + lambda2 + mu2)
tol = 1e4eps()
boundingradius = 4.5
maxiter = 50
CFL = 0.25
angularposition = angular_position(levelsetcoords .- interfacecenter)
sortidx = sortperm(angularposition)
angularposition = angularposition[sortidx]


cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
refseedpoints, refseedcellids =
    CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
spatialseedpoints =
    CutCellDG.map_to_spatial(refseedpoints, refseedcellids, cgmesh)
spatialseedradius =
    vec(mapslices(norm, spatialseedpoints .- interfacecenter, dims = 1))
R0 = TES.average(spatialseedradius)
devR0 = std(spatialseedradius)
analyticalsolution = PS.CylindricalSolver(
    R0,
    outerradius,
    interfacecenter,
    lambda1,
    mu1,
    lambda2,
    mu2,
    theta0,
)
pd0 = TES.potential_difference_at_nodal_coordinates(
    cutmesh,
    elasticitybasis,
    levelset,
    refseedpoints,
    refseedcellids,
    spatialseedpoints,
    stiffness,
    theta0,
    V01,
    V02,
    ΔG0,
    numqp,
    penalty,
    analyticalsolution,
    tol,
    boundingradius,
    maxiter,
)
sortedpd0 = pd0[sortidx]
exactpd0 =
    exact_normalized_potential_difference(analyticalsolution, V01, V02, ΔG0)
errpd0 = potential_error(pd0, exactpd0)



# Δylim = 0.1 * abs(exactpd1)
# fig, ax = PyPlot.subplots()
# ax.plot(angularposition, sortedpd0)
# ax.plot(angularposition, exactpd0 * ones(length(angularposition)))
# ax.grid()
# ax.set_ylim(exactpd0 - Δylim, exactpd0 + Δylim)
# fig









coeffs1 = TES.step_levelset(
    levelset,
    pd0,
    refseedpoints,
    refseedcellids,
    spatialseedpoints,
    tol,
    boundingradius,
    CFL,
)

# fig,ax = PyPlot.subplots()
# ax.tricontour(levelsetcoords[1,:],levelsetcoords[2,:],coeffs0,[0.0])
# ax.tricontour(levelsetcoords[1,:],levelsetcoords[2,:],coeffs1,[0.0])
# ax.grid()
# ax.set_aspect("equal")
# fig


CutCellDG.update_coefficients!(levelset, coeffs1)
cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
refseedpoints, refseedcellids =
    CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
spatialseedpoints =
    CutCellDG.map_to_spatial(refseedpoints, refseedcellids, cutmesh)
spatialseedradius =
    vec(mapslices(norm, spatialseedpoints .- interfacecenter, dims = 1))
oldR1 = TES.average(spatialseedradius)
olddevR1 = std(spatialseedradius)


signeddistance = CutCellDG.distance_to_zero_levelset(
    levelsetcoords,
    refseedpoints,
    spatialseedpoints,
    refseedcellids,
    levelset,
    tol,
    boundingradius,
)
CutCellDG.update_coefficients!(levelset, signeddistance)
refseedpoints, refseedcellids =
    CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
spatialseedpoints =
    CutCellDG.map_to_spatial(refseedpoints, refseedcellids, cutmesh)
spatialseedradius =
    vec(mapslices(norm, spatialseedpoints .- interfacecenter, dims = 1))
R1 = TES.average(spatialseedradius)
devR1 = std(spatialseedradius)


analyticalsolution = PS.CylindricalSolver(
    R1,
    outerradius,
    interfacecenter,
    lambda1,
    mu1,
    lambda2,
    mu2,
    theta0,
)
pd1 = TES.potential_difference_at_nodal_coordinates(
    cutmesh,
    elasticitybasis,
    levelset,
    refseedpoints,
    refseedcellids,
    spatialseedpoints,
    stiffness,
    theta0,
    V01,
    V02,
    ΔG0,
    numqp,
    penalty,
    analyticalsolution,
    tol,
    boundingradius,
    maxiter,
)
sortedpd1 = pd1[sortidx]

exactpd1 =
    exact_normalized_potential_difference(analyticalsolution, V01, V02, ΔG0)
errpd1 = potential_error(pd1, exactpd1)


Δylim = 0.1 * abs(exactpd1)
fig, ax = PyPlot.subplots()
ax.plot(angularposition, sortedpd1)
ax.plot(angularposition, exactpd1 * ones(length(angularposition)))
ax.grid()
ax.set_ylim(exactpd0 - Δylim, exactpd0 + Δylim)
fig



coeffs2 = TES.step_levelset(
    levelset,
    pd1,
    refseedpoints,
    refseedcellids,
    spatialseedpoints,
    tol,
    boundingradius,
    CFL,
)
CutCellDG.update_coefficients!(levelset, coeffs2)
cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
refseedpoints, refseedcellids =
    CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
spatialseedpoints =
    CutCellDG.map_to_spatial(refseedpoints, refseedcellids, cutmesh)
spatialseedradius =
    vec(mapslices(norm, spatialseedpoints .- interfacecenter, dims = 1))
signeddistance = CutCellDG.distance_to_zero_levelset(
    levelsetcoords,
    refseedpoints,
    spatialseedpoints,
    refseedcellids,
    levelset,
    tol,
    boundingradius,
)



R2 = TES.average(spatialseedradius)
devR2 = std(spatialseedradius)
analyticalsolution = PS.CylindricalSolver(
    R2,
    outerradius,
    interfacecenter,
    lambda1,
    mu1,
    lambda2,
    mu2,
    theta0,
)
pd2 = TES.potential_difference_at_nodal_coordinates(
    cutmesh,
    elasticitybasis,
    levelset,
    refseedpoints,
    refseedcellids,
    spatialseedpoints,
    stiffness,
    theta0,
    V01,
    V02,
    ΔG0,
    numqp,
    penalty,
    analyticalsolution,
    tol,
    boundingradius,
    maxiter,
)










#
#
#
#
# Δylim = 0.1*abs(exactpd0)
# fig,ax = PyPlot.subplots()
# ax.plot(angularposition,sortedpd0)
# ax.plot(angularposition,exactpd0*ones(length(angularposition)))
# ax.grid()
# ax.set_ylim(exactpd0 - Δylim, exactpd0 + Δylim)
# fig
