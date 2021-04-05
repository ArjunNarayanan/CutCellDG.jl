using Statistics
using LinearAlgebra
using PyPlot
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("../../../test/useful_routines.jl")
include("transformation-elasticity-solver.jl")

function lame_lambda(k, m)
    return k - 2m / 3
end

function bulk_modulus(l, m)
    return l + 2m / 3
end

function angular_position(points)
    cpoints = points[1, :] + im * points[2, :]
    return rad2deg.(angle.(cpoints))
end

polyorder = 3
nelmts = 33
penaltyfactor = 1e2

folderpath = "examples/potential/cube/"
filename =
    "polyorder-" * string(polyorder) * "-nelmts-" * string(nelmts) * ".png"

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

ΔG0 = -14351.0/0.147


theta0 = -0.067
transfstress =
    CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)

meshwidth = [width, width]
numqp = required_quadrature_order(polyorder) + 2

dx = width / nelmts
penalty = penaltyfactor / dx * 0.5 * (lambda1 + mu1 + lambda2 + mu2)

basis = TensorProductBasis(2, polyorder)
corner = [0.8,0.8]

mesh, cellquads, facequads, interfacequads = construct_mesh_and_quadratures(
    meshwidth,
    nelmts,
    basis,
    corner,
    numqp,
)

nodaldisplacement = free_slip_bc_nodal_displacement(
    mesh,
    basis,
    cellquads,
    facequads,
    interfacequads,
    stiffness,
    theta0,
    penalty,
)

refseedpoints, spatialseedpoints, seedcellids =
    CutCellDG.seed_zero_levelset_with_interfacequads(interfacequads, mesh)
normals = CutCellDG.collect_interface_normals(interfacequads, mesh)

spatialpoints = spatialseedpoints[1, :, :]
productreferencepoints = refseedpoints[1, :, :]
parentreferencepoints = refseedpoints[2, :, :]
productreferencecellids = seedcellids[1, :]
parentreferencecellids = seedcellids[2, :]

angularposition = angular_position(spatialpoints)
sortidx = sortperm(angularposition)

parentreferencepoints = parentreferencepoints[:, sortidx]
productreferencepoints = productreferencepoints[:, sortidx]
parentreferencecellids = parentreferencecellids[sortidx]
productreferencecellids = productreferencecellids[sortidx]
spatialpoints = spatialpoints[:, sortidx]
normals = normals[:, sortidx]
angularposition = angularposition[sortidx]

parentstrain = parent_strain_at_reference_points(
    nodaldisplacement,
    basis,
    parentreferencepoints,
    parentreferencecellids,
    mesh,
)
parentstress = parent_stress(parentstrain, stiffness)

productstrain = product_strain_at_reference_points(
    nodaldisplacement,
    basis,
    theta0,
    productreferencepoints,
    productreferencecellids,
    mesh,
)
productstress = product_stress(productstrain, stiffness, theta0)

parentstrainenergy = V02 * strain_energy(parentstress, parentstrain)
productstrainenergy = V01 * strain_energy(productstress, productstrain)

parentradialtraction = CutCellDG.traction_force_at_points(parentstress, normals)
parentsrr = CutCellDG.traction_component(parentradialtraction, normals)

productradialtraction =
    CutCellDG.traction_force_at_points(productstress, normals)
productsrr = CutCellDG.traction_component(productradialtraction, normals)

parentdilatation = CutCellDG.dilatation(parentstrain)
productdilatation = CutCellDG.dilatation(productstrain)

parentcompwork = V02 * (1 .+ parentdilatation) .* parentsrr
productcompwork = V01 * (1 .+ productdilatation) .* productsrr

parentpotential = parentstrainenergy - parentcompwork
productpotential = productstrainenergy - productcompwork

potentialdifference = (productpotential - parentpotential) * 1e9 / abs(ΔG0)


fig, ax = PyPlot.subplots()
ax.plot(angularposition, potentialdifference, color = "black")
ax.grid()
ax.set_ylim(0.0,0.45)
ax.set_xlabel("Angular Position (deg)")
ax.set_ylabel(L"([\Phi] - [G0])/[G0]")
fig.tight_layout()
fig
fig.savefig(folderpath * filename)
