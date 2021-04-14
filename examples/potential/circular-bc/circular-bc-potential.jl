using PyPlot
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("../../../test/useful_routines.jl")
include("analytical-solver.jl")
PS = PlaneStrainSolver
include("transformation-elasticity-solver.jl")

function lame_lambda(k, m)
    return k - 2m / 3
end

function angular_position(points)
    cpoints = points[1, :] + im * points[2, :]
    return rad2deg.(angle.(cpoints))
end

polyorder = 3
nelmts = 17
penaltyfactor = 1e2

folderpath = "examples/potential/circular-bc/"
filename =
    "polyorder-" * string(polyorder) * "-nelmts-" * string(nelmts) * ".png"

width = 1.0              # mm
K1, K2 = 247.0, 192.0    # GPa
mu1, mu2 = 126.0, 87.0   # GPa
# K1, K2 = 247.0, 247.0
# mu1,mu2 = 126.0, 87.0
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
    construct_mesh_and_quadratures(
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

cutmesh = CutCellDG.background_mesh(mesh)
refseedpoints, refseedcellids =
    CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
spatialpoints = CutCellDG.map_to_spatial(
    refseedpoints[1, :, :],
    refseedcellids[1, :],
    CutCellDG.background_mesh(mesh),
)

normals = CutCellDG.collect_interface_normals(interfacequads, mesh)


productreferencepoints = refseedpoints[1, :, :]
parentreferencepoints = refseedpoints[2, :, :]
productreferencecellids = refseedcellids[1, :]
parentreferencecellids = refseedcellids[2, :]

relspatialpoints = spatialpoints .- interfacecenter
angularposition = angular_position(relspatialpoints)
sortidx = sortperm(angularposition)

angularposition = angularposition[sortidx]
parentreferencepoints = parentreferencepoints[:, sortidx]
productreferencepoints = productreferencepoints[:, sortidx]
parentreferencecellids = parentreferencecellids[sortidx]
productreferencecellids = productreferencecellids[sortidx]
spatialpoints = spatialpoints[:, sortidx]
normals = normals[:, sortidx]

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

exactparentstrainenergy = PS.core_strain_energy(analyticalsolution, V02)
exactproductstrainenergy =
    PS.shell_strain_energy(analyticalsolution, interfaceradius, V01)

parentradialtraction = CutCellDG.traction_force_at_points(parentstress, normals)
parentsrr = CutCellDG.traction_component(parentradialtraction, normals)

productradialtraction =
    CutCellDG.traction_force_at_points(productstress, normals)
productsrr = CutCellDG.traction_component(productradialtraction, normals)

parentdilatation = CutCellDG.dilatation(parentstrain)
productdilatation = CutCellDG.dilatation(productstrain)

parentcompwork = V02 * (1 .+ parentdilatation) .* parentsrr
productcompwork = V01 * (1 .+ productdilatation) .* productsrr

exactparentcompwork = PS.core_compression_work(analyticalsolution, V02)
exactproductcompwork =
    PS.shell_compression_work(analyticalsolution, interfaceradius, V01)

parentpotential = parentstrainenergy - parentcompwork
productpotential = productstrainenergy - productcompwork

exactparentpotential = exactparentstrainenergy - exactparentcompwork
exactproductpotential = exactproductstrainenergy - exactproductcompwork

potentialdifference = (productpotential - parentpotential) * 1e9 / abs(ΔG0)

exactpotentialdifference =
    (exactproductpotential - exactparentpotential) * 1e9 / abs(ΔG0)


pderr =
    maximum(abs.(potentialdifference .- exactpotentialdifference)) /
    abs(exactpotentialdifference)



# Δylim = 0.2 * abs(exactpotentialdifference)
# ylim = (exactpotentialdifference - Δylim, exactpotentialdifference + Δylim)
#
# fig, ax = PyPlot.subplots()
# ax.plot(angularposition, potentialdifference, color = "black")
# ax.plot(
#     angularposition,
#     exactpotentialdifference * ones(length(angularposition)),
#     linestyle = "dotted",
#     color = "black",
# )
# ax.grid()
# ax.set_ylim(ylim...)
# ax.set_xlabel("Angular Position (deg)")
# ax.set_ylabel(L"([\Phi] - [G0])/[G0]")
# fig.tight_layout()
# fig
# fig.savefig(folderpath * filename)

# fig,ax = PyPlot.subplots()
# numplotpts = length(angularposition)
# ax.plot(angularposition,parentpotential)
# ax.plot(angularposition,exactparentpotential*ones(numplotpts))
# ax.grid()
# fig
#
# fig,ax = PyPlot.subplots()
# numplotpts = length(angularposition)
# ax.plot(angularposition,productpotential)
# ax.plot(angularposition,exactproductpotential*ones(numplotpts))
# ax.grid()
# fig

# fig,ax = PyPlot.subplots()
# numplotpts = length(angularposition)
# ax.plot(angularposition,productstrainenergy)
# ax.plot(angularposition,exactproductstrainenergy*ones(numplotpts))
# ax.grid()
# fig

# fig,ax = PyPlot.subplots()
# numplotpts = length(angularposition)
# ax.plot(angularposition,productcompwork)
# ax.plot(angularposition,exactproductcompwork*ones(numplotpts))
# ax.grid()
# fig


# productpressure = CutCellDG.pressure_at_points(productstress)
# exactproductpressure = PS.shell_pressure(analyticalsolution,interfaceradius)

########################################################################
# productpressure = pressure(productstress)
# parentpressure = pressure(parentstress)
# exactparentpressure = -1/3*sum(core_stress(analyticalsolution))
# exactproductpressure = -1/3*sum(shell_stress(analyticalsolution,interfaceradius))
# fig,ax = PyPlot.subplots(2,1)
# numplotpts = length(angularposition)
# ax[1].plot(angularposition,productpressure)
# ax[1].plot(angularposition,exactproductpressure*ones(numplotpts),"--")
# ax[1].grid()
# ax[1].set_ylim(-3.0,-2.5)
# ax[2].plot(angularposition,parentpressure)
# ax[2].plot(angularposition,exactparentpressure*ones(numplotpts),"--")
# ax[2].grid()
# ax[2].set_ylim(3.25,3.75)
# fig
########################################################################
