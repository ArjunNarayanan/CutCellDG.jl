using PyPlot
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("../../../test/useful_routines.jl")
include("../elasticity-solver-uniform-pressure.jl")

function angular_position(points)
    cpoints = points[1, :] + im * points[2, :]
    return rad2deg.(angle.(cpoints))
end


K1, K2 = 247.0, 247.0    # Pa
mu1, mu2 = 126.0, 126.0   # Pa
lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)
stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)
penaltyfactor = 1e2
eta = 1

polyorder = 2
numqp = required_quadrature_order(polyorder) + 2
nelmts = 17
basis = TensorProductBasis(2, polyorder)
interface_center = [0.5, 0.5]
interface_radius = 1.0/3.0

distancefunc(x) = -circle_distance_function(x,interface_center,interface_radius)

nodaldisplacement, mergedmesh, cellquads, facequads, interfacequads =
    nodal_displacement(
        distancefunc,
        stiffness,
        nelmts,
        basis,
        numqp,
        penaltyfactor,
        eta,
    )

refseedpoints, spatialseedpoints, seedcellids =
    CutCellDG.seed_zero_levelset_with_interfacequads(interfacequads, mergedmesh)
interfacenormals =
    CutCellDG.collect_interface_normals(interfacequads, mergedmesh)

spatialpoints = spatialseedpoints[1, :, :]
referencepoints = refseedpoints
referencecellids = seedcellids

relspatialpoints = spatialpoints .- interface_center
angularposition = angular_position(relspatialpoints)
sortidx = sortperm(angularposition)

angularposition = angularposition[sortidx]
spatialpoints = spatialpoints[:, sortidx]
referencepoints = referencepoints[:, :, sortidx]
referencecellids = referencecellids[:, sortidx]
interfacenormals = interfacenormals[:, sortidx]


productdisplacement = CutCellDG.displacement_at_reference_points(
    nodaldisplacement,
    basis,
    referencepoints,
    referencecellids,
    +1,
    mergedmesh,
)
parentdisplacement = CutCellDG.displacement_at_reference_points(
    nodaldisplacement,
    basis,
    referencepoints,
    referencecellids,
    -1,
    mergedmesh,
)
# exactdisplacement =
#     mapslices(x -> displacement(0.1, x), spatialpoints, dims = 1)


fig,ax = PyPlot.subplots(2,1)
ax[1].plot(angularposition,productdisplacement[1,:],label="product u1")
ax[1].plot(angularposition,parentdisplacement[1,:],label="parent u1")
# ax[1].plot(angularposition,exactdisplacement[1,:],"--",label="exact")
ax[2].plot(angularposition,productdisplacement[2,:],label="product u1")
ax[2].plot(angularposition,parentdisplacement[2,:],label="parent u1")
# ax[2].plot(angularposition,exactdisplacement[2,:],"--",label="exact")
ax[1].grid()
ax[2].grid()
ax[1].legend()
ax[2].legend()
fig


productdisplacementmagnitude = vec(mapslices(norm,productdisplacement,dims=1))
parentdisplacementmagnitude = vec(mapslices(norm,parentdisplacement,dims=1))

diffproduct = maximum(productdisplacementmagnitude) - minimum(productdisplacementmagnitude)
diffparent = maximum(parentdisplacementmagnitude) - minimum(parentdisplacementmagnitude)


fig,ax = PyPlot.subplots()
ax.plot(angularposition,productdisplacementmagnitude,label="product")
ax.plot(angularposition,parentdisplacementmagnitude,label="parent")
ax.legend()
ax.grid()
fig



productstress = CutCellDG.stress_at_reference_points(
    nodaldisplacement,
    basis,
    stiffness,
    referencepoints,
    referencecellids,
    +1,
    mergedmesh,
)
producttraction = CutCellDG.traction_force_at_points(productstress,interfacenormals)

parentstress = CutCellDG.stress_at_reference_points(
    nodaldisplacement,
    basis,
    stiffness,
    referencepoints,
    referencecellids,
    -1,
    mergedmesh,
)
parenttraction = CutCellDG.traction_force_at_points(parentstress,interfacenormals)
#
# exactstress = mapslices(x->stress_field(lambda1,mu1,0.1,x),spatialpoints,dims=1)
# exacttraction = CutCellDG.traction_force_at_points(exactstress,interfacenormals)
#
#
# fig,ax = PyPlot.subplots(2,1)
#
# ax[1].plot(angularposition,producttraction[1,:],label="product t1")
# ax[1].plot(angularposition,parenttraction[1,:],label="parent t1")
# ax[1].plot(angularposition,exacttraction[1,:],"--",label="exact t1")
# ax[2].plot(angularposition,producttraction[2,:],label="product t2")
# ax[2].plot(angularposition,parenttraction[2,:],label="parent t2")
# ax[2].plot(angularposition,exacttraction[2,:],"--",label="exact t2")
#
# ax[1].grid()
# ax[2].grid()
# ax[1].legend()
# ax[2].legend()
#
# fig


producttractionmagnitude = vec(mapslices(norm,producttraction,dims=1))
parenttractionmagnitude = vec(mapslices(norm,parenttraction,dims=1))

fig,ax = PyPlot.subplots()
ax.plot(angularposition,producttractionmagnitude)
ax.plot(angularposition,parenttractionmagnitude)
ax.grid()
fig
