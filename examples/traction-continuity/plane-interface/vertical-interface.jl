using PyPlot
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("../../../test/useful_routines.jl")
include("../elasticity-solver.jl")

K1, K2 = 247.0, 192.0    # Pa
mu1, mu2 = 126.0, 87.0   # Pa
lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)
stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)
penaltyfactor = 1e2
eta = 1

polyorder = 2
numqp = required_quadrature_order(polyorder) + 2
nelmts = 17
basis = TensorProductBasis(2, polyorder)
interfacepoint = [0.8, 0.0]
interfaceangle = 40.0
interfacenormal = [cosd(interfaceangle), sind(interfaceangle)]

distancefunc(x) = plane_distance_function(x, interfacenormal, interfacepoint)

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


sortidx = sortperm(spatialpoints[2, :])
spatialpoints = spatialpoints[:, sortidx]
referencepoints = referencepoints[:, :, sortidx]
referencecellids = referencecellids[:, sortidx]
interfacenormals = interfacenormals[:, sortidx]

spycoords = spatialpoints[2, :]

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
exactdisplacement =
    mapslices(x -> displacement(0.1, x), spatialpoints, dims = 1)

# using PyPlot
# fig,ax = PyPlot.subplots(2,1)
# ax[1].plot(spycoords,productdisplacement[1,:],label="product u1")
# ax[1].plot(spycoords,parentdisplacement[1,:],label="parent u1")
# ax[1].plot(spycoords,exactdisplacement[1,:],"--",label="exact")
# ax[2].plot(spycoords,productdisplacement[2,:],label="product u1")
# ax[2].plot(spycoords,parentdisplacement[2,:],label="parent u1")
# ax[2].plot(spycoords,exactdisplacement[2,:],"--",label="exact")
# ax[1].grid()
# ax[2].grid()
# ax[1].legend()
# ax[2].legend()
# fig


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

exactstress = mapslices(x->stress_field(lambda1,mu1,0.1,x),spatialpoints,dims=1)
exacttraction = CutCellDG.traction_force_at_points(exactstress,interfacenormals)


fig,ax = PyPlot.subplots(2,1)

ax[1].plot(spycoords,producttraction[1,:],label="product t1")
ax[1].plot(spycoords,parenttraction[1,:],label="parent t1")
ax[1].plot(spycoords,exacttraction[1,:],"--",label="exact t1")

ax[2].plot(spycoords,producttraction[2,:],label="product t2")
ax[2].plot(spycoords,parenttraction[2,:],label="parent t2")
ax[2].plot(spycoords,exacttraction[2,:],"--",label="exact t2")

ax[1].grid()
ax[2].grid()
ax[1].legend()
ax[2].legend()

fig
