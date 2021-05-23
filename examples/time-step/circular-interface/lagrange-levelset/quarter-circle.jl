using PyPlot
using Statistics
using LinearAlgebra
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

function normalized_maxnorm_error(pd, exactpd)
    return maximum(abs.(pd .- exactpd)) / abs(exactpd)
end

function exact_normalized_potential_difference(solver, V01, V02, ΔG0)
    return (ΔG0 + 1e9PS.interface_potential_difference(solver, V01, V02)) /
           abs(ΔG0)
end

function signed_distance_coefficients(levelset, mesh, tol, boundingradius)
    cutmesh = CutCellDG.CutMesh(mesh, levelset)
    refseedpoints, refseedcellids =
        CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
    spatialseedpoints =
        CutCellDG.map_to_spatial(refseedpoints, refseedcellids, mesh)
    nodalcoordinates = CutCellDG.nodal_coordinates(mesh)

    signeddistance = CutCellDG.distance_to_zero_levelset(
        nodalcoordinates,
        spatialseedpoints,
        refseedcellids,
        levelset,
        tol,
        boundingradius,
    )

    return signeddistance
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

basis = TensorProductBasis(2, polyorder)
interfacecenter = [0.0, 0.0]
interfaceradius = 0.8
outerradius = 1.5

cgmesh = CutCellDG.CGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basis)
dgmesh = CutCellDG.DGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basis)
levelset = CutCellDG.LevelSet(
    x -> -circle_distance_function(x, interfacecenter, interfaceradius),
    cgmesh,
    basis,
)
paddedmesh = CutCellDG.BoundaryPaddedMesh(cgmesh, 1)

nodalcoordinates = CutCellDG.nodal_coordinates(cgmesh)
elementsize = CutCellDG.element_size(cgmesh)
minelmtsize = minimum(elementsize)
maxelmtsize = maximum(elementsize)
penalty = penaltyfactor / minelmtsize * 0.5 * (lambda1 + mu1 + lambda2 + mu2)
tol = minelmtsize^(polyorder + 1)
boundingradius = 1.5 * maxelmtsize

angularposition = angular_position(nodalcoordinates .- interfacecenter)
sortidx = sortperm(angularposition)
angularposition = angularposition[sortidx]

mingridsize = minimum(CutCellDG.grid_size(paddedmesh))
CFL = 0.5

coeffs0 = CutCellDG.coefficients(levelset)
CutCellDG.update_coefficients!(levelset, coeffs0)
cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
refseedpoints, seedcellids = CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
spatialseedpoints =
    CutCellDG.map_to_spatial(refseedpoints, seedcellids, cutmesh)
spatialseedradius =
    vec(mapslices(norm, spatialseedpoints .- interfacecenter, dims = 1))
R0 = mean(spatialseedradius)
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
    basis,
    levelset,
    stiffness,
    theta0,
    analyticalsolution,
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
exactpd0 = exact_normalized_potential_difference(analyticalsolution,V01,V02,ΔG0)
err0 = normalized_maxnorm_error(pd0,exactpd0)


coeffs1 = TES.step_levelset(
    levelset,
    pd0,
    spatialseedpoints,
    seedcellids,
    paddedmesh,
    tol,
    boundingradius,
    CFL,
)
CutCellDG.update_coefficients!(levelset, coeffs1)
cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
refseedpoints, seedcellids = CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
spatialseedpoints =
    CutCellDG.map_to_spatial(refseedpoints, seedcellids, cutmesh)
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
    basis,
    levelset,
    stiffness,
    theta0,
    analyticalsolution,
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

exactpd1 = exact_normalized_potential_difference(analyticalsolution,V01,V02,ΔG0)
err1 = normalized_maxnorm_error(pd1,exactpd1)



Δylim = 0.1*abs(exactpd1)
sortedpd1 = pd1[sortidx]
fig,ax = PyPlot.subplots()
ax.plot(angularposition,sortedpd1)
ax.plot(angularposition,exactpd1*ones(length(angularposition)))
ax.grid()
ax.set_ylim(exactpd1-Δylim,exactpd1+Δylim)
fig


# fig, ax = PyPlot.subplots()
# ax.tricontour(nodalcoordinates[1, :], nodalcoordinates[2, :], coeffs0, [0.0])
# ax.tricontour(nodalcoordinates[1, :], nodalcoordinates[2, :], coeffs1, [0.0])
# ax.set_aspect("equal")
# ax.grid()
# fig
