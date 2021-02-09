using PyPlot
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
nelmts = 17
polyorder = 2
numqp = required_quadrature_order(polyorder) + 2
penaltyfactor = 1e2

dx = minimum(meshwidth) / nelmts
penalty = penaltyfactor / dx * 0.5 * (lambda1 + mu1 + lambda2 + mu2)

basis = TensorProductBasis(2, polyorder)
interfacecenter = [0.5, 0.5]
interfaceradius = 1.0 / 3.0
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


productdisplacement = CutCellDG.displacement_at_reference_points(
    nodaldisplacement,
    basis,
    referencepoints,
    referencecellids,
    +1,
    mesh,
)
parentdisplacement = CutCellDG.displacement_at_reference_points(
    nodaldisplacement,
    basis,
    referencepoints,
    referencecellids,
    -1,
    mesh,
)
exactdisplacement = mapslices(analyticalsolution, spatialpoints, dims = 1)



# fig,ax = PyPlot.subplots(2,1)
# ax[1].plot(angularposition,productdisplacement[1,:],label="product u1")
# ax[1].plot(angularposition,parentdisplacement[1,:],label="parent u1")
# ax[1].plot(angularposition,exactdisplacement[1,:],"--",label="exact")
# ax[2].plot(angularposition,productdisplacement[2,:],label="product u1")
# ax[2].plot(angularposition,parentdisplacement[2,:],label="parent u1")
# ax[2].plot(angularposition,exactdisplacement[2,:],"--",label="exact")
# ax[1].grid()
# ax[2].grid()
# ax[1].legend()
# ax[2].legend()
# fig


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


producttraction = CutCellDG.traction_force_at_points(productstress,normals)
parenttraction = CutCellDG.traction_force_at_points(parentstress,normals)

# fig, ax = PyPlot.subplots(2, 1)
# ax[1].plot(angularposition, producttraction[1, :], label = "product")
# ax[1].plot(angularposition, parenttraction[1, :], label = "parent")
# # ax[1].plot(angularposition, exactproducttraction[1, :], "--", label = "exact product")
# # ax[1].plot(angularposition, exactparenttraction[1, :], "--", label = "exact parent")
# ax[1].set_title("t1")
# ax[1].legend()
# ax[1].grid()
# ax[2].plot(angularposition, producttraction[2, :], label = "product")
# ax[2].plot(angularposition, parenttraction[2, :], label = "parent")
# # ax[2].plot(angularposition, exactproducttraction[2, :], "--", label = "exact product")
# # ax[2].plot(angularposition, exactparenttraction[2, :], "--", label = "exact parent")
# ax[2].set_title("t2")
# ax[2].legend()
# ax[2].grid()
# fig.tight_layout()
# fig


productpressure = CutCellDG.pressure_at_points(productstress)
parentpressure = CutCellDG.pressure_at_points(parentstress)

fig,ax = PyPlot.subplots()
ax.plot(angularposition,parentpressure)
fig
