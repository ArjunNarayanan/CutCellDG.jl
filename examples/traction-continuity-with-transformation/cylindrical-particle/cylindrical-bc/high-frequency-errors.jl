using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("../../../../test/useful_routines.jl")
include("transformation-elasticity-solver.jl")

function angular_position(points)
    cpoints = points[1, :] + im * points[2, :]
    return rad2deg.(angle.(cpoints))
end


K1, K2 = 247.0, 192.0    # Pa
mu1, mu2 = 126.0, 87.0   # Pa
lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)
stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)

theta0 = -0.067
stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)
transfstress =
    CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)

meshwidth = [1.0, 1.0]
delta = 1e-1meshwidth[1]
nelmts = 3
polyorder = 3
numqp = required_quadrature_order(polyorder) + 2
highresnumqp = 50
penaltyfactor = 1e3

folderpath = "examples/traction-continuity-with-transformation/cylindrical-particle/cylindrical-bc/"
filename = "equal-moduli-pure-pressure-high-resolution"

dx = minimum(meshwidth) / nelmts
penalty = penaltyfactor / dx * 0.5 * (lambda1 + mu1 + lambda2 + mu2)

basis = TensorProductBasis(2, polyorder)
interfacecenter = [0.5, 0.5]
interfaceradius = dx / sqrt(2) + delta
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
    p0,
)

mesh, cellquads, facequads, interfacequads = construct_mesh_and_quadratures(
    meshwidth,
    nelmts,
    basis,
    interfacecenter,
    interfaceradius,
    numqp,
)

mesh2, cellquads2, facequads2, highresinterfacequads =
    construct_mesh_and_quadratures(
        meshwidth,
        nelmts,
        basis,
        interfacecenter,
        interfaceradius,
        highresnumqp,
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

L2error = mesh_L2_error(
    reshape(nodaldisplacement, 2, :),
    analyticalsolution,
    basis,
    cellquads,
    mesh,
)
exactsolutionL2error =
    integral_norm_on_mesh(analyticalsolution, cellquads, mesh, 2)
normalizedL2error = L2error ./ exactsolutionL2error

refseedpoints, spatialseedpoints, seedcellids =
    CutCellDG.seed_zero_levelset_with_interfacequads(
        highresinterfacequads,
        mesh,
    )
# checkcellids = 8
# refseedpoints, spatialseedpoints, seedcellids =
#     CutCellDG.seed_zero_levelset_with_interfacequads(
#         highresinterfacequads,
#         mesh,
#         checkcellids,
#     )


normals = CutCellDG.collect_interface_normals(highresinterfacequads, mesh)
# normals = CutCellDG.collect_interface_normals(highresinterfacequads,mesh,checkcellids)

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


productdisplacement = CutCellDG.displacement_at_reference_points(
    nodaldisplacement,
    basis,
    referencepoints[1,:,:],
    referencecellids[1,:],
    +1,
    mesh,
)
parentdisplacement = CutCellDG.displacement_at_reference_points(
    nodaldisplacement,
    basis,
    referencepoints[2,:,:],
    referencecellids[2,:],
    -1,
    mesh,
)
exactdisplacement = mapslices(analyticalsolution, spatialpoints, dims = 1)

using PyPlot

# fig,ax = PyPlot.subplots(2,1)
# ax[1].plot(angularposition,productdisplacement[1,:],label="product u1")
# ax[1].plot(angularposition,parentdisplacement[1,:],label="parent u1")
# ax[1].plot(angularposition,exactdisplacement[1,:],"--",label="exact")
# ax[2].plot(angularposition,productdisplacement[2,:],label="product u2")
# ax[2].plot(angularposition,parentdisplacement[2,:],label="parent u2")
# ax[2].plot(angularposition,exactdisplacement[2,:],"--",label="exact")
# ax[1].grid()
# ax[2].grid()
# ax[1].legend()
# ax[2].legend()
# fig

productdisplacementmagnitude =
    vec(mapslices(norm, productdisplacement, dims = 1))
parentdisplacementmagnitude = vec(mapslices(norm, parentdisplacement, dims = 1))
exactdisplacementmagnitude = vec(mapslices(norm, exactdisplacement, dims = 1))

parentdisplacementerror =
    abs.(parentdisplacementmagnitude - exactdisplacementmagnitude) ./
    exactdisplacementmagnitude
productdisplacementerror =
    abs.(productdisplacementmagnitude - exactdisplacementmagnitude) ./
    exactdisplacementmagnitude

fig, ax = PyPlot.subplots()
ax.plot(angularposition, parentdisplacementerror, label = "parent")
ax.plot(angularposition, productdisplacementerror, label = "product")
ax.legend()
ax.grid()
# ax.set_ylim(0, 7e-4)
# ax.set_xlim(0, 90)
# ax.set_title("DG + Merging: Displacement Error")
fig
# fig.savefig(folderpath*filename*"-cell-"*string(checkcellids)*"-displacement-error.png")


productstress = product_stress_at_reference_points(
    nodaldisplacement,
    basis,
    stiffness,
    transfstress,
    theta0,
    referencepoints,
    referencecellids,
    mesh,
)
parentstress = parent_stress_at_reference_points(
    nodaldisplacement,
    basis,
    stiffness,
    referencepoints,
    referencecellids,
    mesh,
)
exactstress =
    mapslices(x -> exact_stress(analyticalsolution, x), spatialpoints, dims = 1)

producttraction = CutCellDG.traction_force_at_points(productstress, normals)
parenttraction = CutCellDG.traction_force_at_points(parentstress, normals)
exacttraction = CutCellDG.traction_force_at_points(exactstress, normals)

# fig, ax = PyPlot.subplots(2, 1)
# ax[1].plot(angularposition, producttraction[1, :], label = "product")
# ax[1].plot(angularposition, parenttraction[1, :], label = "parent")
# ax[1].plot(angularposition, exacttraction[1, :], "--", label = "exact")
# ax[1].set_title("t1")
# ax[1].legend()
# ax[1].grid()
# ax[2].plot(angularposition, producttraction[2, :], label = "product")
# ax[2].plot(angularposition, parenttraction[2, :], label = "parent")
# ax[2].plot(angularposition, exacttraction[2, :], "--", label = "exact")
# ax[2].set_title("t2")
# ax[2].legend()
# ax[2].grid()
# fig.tight_layout()
# fig


producttractionmagnitude = vec(mapslices(norm, producttraction, dims = 1))
parenttractionmagnitude = vec(mapslices(norm, parenttraction, dims = 1))
exacttractionmagnitude = vec(mapslices(norm, exacttraction, dims = 1))

producttractionerror =
    abs.(producttractionmagnitude - exacttractionmagnitude) ./
    exacttractionmagnitude
parenttractionerror =
    abs.(parenttractionmagnitude - exacttractionmagnitude) ./
    exacttractionmagnitude

fig, ax = PyPlot.subplots()
ax.plot(angularposition, parenttractionerror, label = "parent")
ax.plot(angularposition, producttractionerror, label = "product")
ax.legend()
ax.grid()
# ax.set_xlim(0,90)
# ax.set_ylim(0,0.006)
ax.set_title("DG + Merging: Traction Error")
fig
# fig.savefig(folderpath*filename*"-cell-"*string(checkcellids)*"-traction-error.png")
