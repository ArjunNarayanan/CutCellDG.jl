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
p0 = -0.0
stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)
transfstress =
    CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)

meshwidth = [1.0, 1.0]
delta = 1e-2meshwidth[1]
nelmts = 3
polyorder = 2
numqp = required_quadrature_order(polyorder) + 2
highresnumqp = 10
penaltyfactor = 1e2

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

mesh2, highrescellquads, facequads2, highresinterfacequads =
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


productrefpoints, productsppoints, productrefcellids =
    CutCellDG.collect_cell_quadratures(highrescellquads, mesh, +1)
parentrefpoints, parentsppoints, parentrefcellids =
    CutCellDG.collect_cell_quadratures(highrescellquads, mesh, -1)

productdisplacement = CutCellDG.displacement_at_reference_points(
    nodaldisplacement,
    basis,
    productrefpoints,
    productrefcellids,
    +1,
    mesh,
)
parentdisplacement = CutCellDG.displacement_at_reference_points(
    nodaldisplacement,
    basis,
    parentrefpoints,
    parentrefcellids,
    -1,
    mesh,
)
exactproductdisplacement =
    mapslices(analyticalsolution, productsppoints, dims = 1)
exactparentdisplacement =
    mapslices(analyticalsolution, parentsppoints, dims = 1)



parentdisplacementerror =
    vec(
        mapslices(norm, parentdisplacement - exactparentdisplacement, dims = 1),
    ) ./ vec(mapslices(norm, exactparentdisplacement, dims = 1))

productdisplacementerror = vec(
    mapslices(norm, productdisplacement - exactproductdisplacement, dims = 1),
) ./ vec(mapslices(norm,exactproductdisplacement, dims = 1))


displacementerror = vcat(parentdisplacementerror, productdisplacementerror)
spatialpoints = hcat(parentsppoints, productsppoints)


using PyPlot


fig, ax = PyPlot.subplots()
CS = ax.tricontourf(spatialpoints[1, :], spatialpoints[2, :], displacementerror)
ax.clabel(CS, inline = true)
ax.set_aspect("equal")
fig


# productstress = product_stress_at_reference_points(
#     nodaldisplacement,
#     basis,
#     stiffness,
#     transfstress,
#     theta0,
#     referencepoints,
#     referencecellids,
#     mesh,
# )
# parentstress = parent_stress_at_reference_points(
#     nodaldisplacement,
#     basis,
#     stiffness,
#     referencepoints,
#     referencecellids,
#     mesh,
# )
# exactstress =
#     mapslices(x -> exact_stress(analyticalsolution, x), spatialpoints, dims = 1)
#
# producttraction = CutCellDG.traction_force_at_points(productstress, normals)
# parenttraction = CutCellDG.traction_force_at_points(parentstress, normals)
# exacttraction = CutCellDG.traction_force_at_points(exactstress, normals)
#
# # fig, ax = PyPlot.subplots(2, 1)
# # ax[1].plot(angularposition, producttraction[1, :], label = "product")
# # ax[1].plot(angularposition, parenttraction[1, :], label = "parent")
# # ax[1].plot(angularposition, exacttraction[1, :], "--", label = "exact")
# # ax[1].set_title("t1")
# # ax[1].legend()
# # ax[1].grid()
# # ax[2].plot(angularposition, producttraction[2, :], label = "product")
# # ax[2].plot(angularposition, parenttraction[2, :], label = "parent")
# # ax[2].plot(angularposition, exacttraction[2, :], "--", label = "exact")
# # ax[2].set_title("t2")
# # ax[2].legend()
# # ax[2].grid()
# # fig.tight_layout()
# # fig
#
#
# producttractionmagnitude = vec(mapslices(norm, producttraction, dims = 1))
# parenttractionmagnitude = vec(mapslices(norm, parenttraction, dims = 1))
# exacttractionmagnitude = vec(mapslices(norm, exacttraction, dims = 1))
#
# producttractionerror =
#     abs.(producttractionmagnitude - exacttractionmagnitude) ./
#     exacttractionmagnitude
# parenttractionerror =
#     abs.(parenttractionmagnitude - exacttractionmagnitude) ./
#     exacttractionmagnitude
#
# fig, ax = PyPlot.subplots()
# ax.plot(angularposition, parenttractionerror, label = "parent")
# ax.plot(angularposition, producttractionerror, label = "product")
# ax.legend()
# ax.grid()
# ax.set_xlim(0,90)
# ax.set_ylim(0,0.006)
# ax.set_title("DG + Merging: Traction Error")
# fig
# fig.savefig(folderpath*filename*"-cell-"*string(checkcellids)*"-traction-error.png")
