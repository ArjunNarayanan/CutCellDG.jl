using LinearAlgebra
using PyPlot
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("../../../test/useful_routines.jl")
include("transformation-elasticity-solver.jl")

function angular_position(points)
    cpoints = points[1, :] + im * points[2, :]
    return rad2deg.(angle.(cpoints))
end

polyorder = 3
nelmts = 9
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

ΔG0 = -6.95e-3

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
outerradius = 2.0
analyticalsolution = AnalyticalSolution(
    interfaceradius,
    outerradius,
    interfacecenter,
    lambda1,
    mu1,
    lambda2,
    mu2,
    theta0,
)

mesh, cellquads, facequads, interfacequads = construct_mesh_and_quadratures(
    meshwidth,
    nelmts,
    basis,
    interfacecenter,
    interfaceradius,
    numqp,
)

nodaldisplacement = nodal_displacement(
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

refseedpoints, spatialseedpoints, seedcellids =
    CutCellDG.seed_zero_levelset_with_interfacequads(interfacequads, mesh)
normals = CutCellDG.collect_interface_normals(interfacequads, mesh)

spatialpoints = spatialseedpoints[1, :, :]
referencepoints = refseedpoints
referencecellids = seedcellids

relspatialpoints = spatialpoints .- interfacecenter
angularposition = angular_position(relspatialpoints)
sortidx = sortperm(angularposition)
angularposition = angularposition[sortidx]

referencepoints = referencepoints[:, :, sortidx]
referencecellids = referencecellids[:, sortidx]
spatialpoints = spatialpoints[:, sortidx]
normals = normals[:, sortidx]

productstress = product_stress_at_reference_points(
    nodaldisplacement,
    basis,
    stiffness,
    transfstress,
    theta0,
    referencepoints[1, :, :],
    referencecellids[1, :],
    mesh,
)
parentstress = parent_stress_at_reference_points(
    nodaldisplacement,
    basis,
    stiffness,
    referencepoints[2, :, :],
    referencecellids[2, :],
    mesh,
)

productpressure = pressure(productstress)
parentpressure = pressure(parentstress)

productdevstress = deviatoric_stress(productstress, productpressure)
parentdevstress = deviatoric_stress(parentstress, parentpressure)

productdevtraction =
    CutCellDG.traction_force_at_points(productdevstress, normals)
parentdevtraction = CutCellDG.traction_force_at_points(parentdevstress, normals)

productdevnormalcomponent = vec(sum(productdevtraction .* normals, dims = 1))
parentdevnormalcomponent = vec(sum(parentdevtraction .* normals, dims = 1))

productdevnorm = CutCellDG.stress_inner_product(productdevstress)
parentdevnorm = CutCellDG.stress_inner_product(parentdevstress)

productspecificvolume = V01 * (1.0 .- productpressure / K1)
parentspecificvolume = V02 * (1.0 .- parentpressure / K2)

productpotential1 = productpressure * V01
productpotential2 = -(productpressure .^ 2) * V01 / (2K1)
productpotential3 = -productspecificvolume .* productdevnormalcomponent
productpotential4 = V01 / (4mu1) * productdevnorm

productpotential =
    productpotential1 +
    productpotential2 +
    productpotential3 +
    productpotential4

fig, ax = PyPlot.subplots()
ax.plot(angularposition, productpotential1, label = "p1")
ax.plot(angularposition, productpotential2, label = "p2")
ax.plot(angularposition, productpotential3, label = "p3")
ax.plot(angularposition, productpotential4, label = "p4")
ax.grid()
ax.legend()
fig

parentpotential1 = parentpressure * V02
parentpotential2 = -(parentpressure .^ 2) * V02 / (2K2)
parentpotential3 = -parentspecificvolume .* parentdevnormalcomponent
parentpotential4 = V02 / (4mu2) * parentdevnorm

parentpotential =
    parentpotential1 +
    parentpotential2 +
    parentpotential3 +
    parentpotential4

fig, ax = PyPlot.subplots()
ax.plot(angularposition, parentpotential1, label = "p1")
ax.plot(angularposition, parentpotential2, label = "p2")
ax.plot(angularposition, parentpotential3, label = "p3")
ax.plot(angularposition, parentpotential4, label = "p4")
ax.grid()
ax.legend()
fig

potentialdifference = ΔG0 .+ (productpotential - parentpotential)

fig,ax = PyPlot.subplots()
ax.plot(angularposition,potentialdifference)
ax.grid()
fig
