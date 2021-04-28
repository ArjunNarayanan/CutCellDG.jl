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

function normalized_maxnorm_error(pd, exactpd)
    return maximum(abs.(pd .- exactpd)) / abs(exactpd)
end

function exact_normalized_potential_difference(solver, V01, V02, ΔG0)
    return (ΔG0 + 1e9PS.interface_potential_difference(solver, V01, V02)) /
           abs(ΔG0)
end

polyorder = 3
nelmts = 15
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
interfacecenter = [0.5, 0.5]
interfaceradius = 0.395
outerradius = 1.2
CFL = 0.25


cgmesh = CutCellDG.CGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basis)
dgmesh = CutCellDG.DGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basis)
levelset = CutCellDG.LevelSet(
    x -> -circle_distance_function(x, interfacecenter, interfaceradius),
    cgmesh,
    basis,
)
coeffs0 = copy(CutCellDG.coefficients(levelset))
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



################################################################################
# POTENTIAL DIFFERENCE AT TIME = 0
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
mergedmesh0, cellquads, facequads, interfacequads =
    TES.construct_merged_mesh_and_quadratures(cutmesh, levelset, numqp)

nodaldisplacement = TES.nodal_displacement(
    mergedmesh0,
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
pd0 =
    (
        ΔG0 .+
        1e9 * TES.potential_difference_at_closest_points(
            querypoints,
            nodaldisplacement,
            basis,
            spatialseedpoints,
            seedcellids,
            mergedmesh0,
            levelset,
            stiffness,
            theta0,
            V01,
            V02,
            tol,
            boundingradius,
        )
    ) / abs(ΔG0)
exactpd0 =
    exact_normalized_potential_difference(analyticalsolution, V01, V02, ΔG0)
err0 = abs.(pd0 .- exactpd0)
maxerr0 = maximum(err0)
################################################################################





################################################################################
# POTENTIAL DIFFERENCE AT TIME STEP = 1
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
mergedmesh1, cellquads, facequads, interfacequads =
    TES.construct_merged_mesh_and_quadratures(cutmesh, levelset, numqp)
nodaldisplacement = TES.nodal_displacement(
    mergedmesh1,
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
pd1 =
    (
        ΔG0 .+
        1e9 * TES.potential_difference_at_closest_points(
            querypoints,
            nodaldisplacement,
            basis,
            spatialseedpoints,
            seedcellids,
            mergedmesh1,
            levelset,
            stiffness,
            theta0,
            V01,
            V02,
            tol,
            boundingradius,
        )
    ) / abs(ΔG0)
exactpd1 =
    exact_normalized_potential_difference(analyticalsolution, V01, V02, ΔG0)
err1 = abs.(pd1 .- exactpd1)
maxerr1 = maximum(err1)
################################################################################



CutCellDG.update_coefficients!(levelset,coeffs0)
maxerridx = sortperm(err0)
xq = nodalcoordinates[:, maxerridx[1:20]]
xcp, xcellid = CutCellDG.closest_points_on_zero_levelset(
    xq,
    spatialseedpoints,
    seedcellids,
    levelset,
    tol,
    boundingradius,
)

testcellids = xcellid[1:5]
using Plots
CutCellDG.load_coefficients!(levelset,xcellid[1])
# CutCellDG.load_coefficients!(levelset,testcellids[1])
poly = CutCellDG.interpolater(levelset)
xrange = -1:1e-2:1
Plots.contour(xrange,xrange,(x,y)->poly([x,y]),levels=[0.0])
