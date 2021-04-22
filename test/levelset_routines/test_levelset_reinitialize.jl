using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("../useful_routines.jl")


function reinitialization_error(distancefunction, nelmts, polyorder)
    L, W = 1.0, 1.0
    basis = TensorProductBasis(2, polyorder)
    numqp = required_quadrature_order(polyorder) + 4
    quad = tensor_product_quadrature(2, numqp)

    mesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
    dx = CutCellDG.element_size(mesh)
    levelset = CutCellDG.LevelSet(distancefunction, mesh, basis)

    cutmesh = CutCellDG.CutMesh(mesh, levelset)

    refseedpoints, seedcellids =
        CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
    spatialseedpoints =
        CutCellDG.map_to_spatial(refseedpoints, seedcellids, mesh)

    nodalcoordinates = CutCellDG.nodal_coordinates(cutmesh)
    boundingradius = 1.5*maximum(dx)
    tol = minimum(dx)^polyorder
    signeddistance = CutCellDG.distance_to_zero_levelset(
        nodalcoordinates,
        spatialseedpoints,
        seedcellids,
        levelset,
        tol,
        boundingradius,
    )

    err = uniform_mesh_L2_error(
        signeddistance',
        x -> distancefunction(x)[1],
        basis,
        quad,
        mesh,
    )

    den = integral_norm_on_uniform_mesh(distancefunction, quad, mesh, 1)

    return err[1] / den[1]
end


xc = [0.5, 0.5]
rad = 0.25
polyorder = 2
powers = [2, 3, 4, 5]
nelmts = [2^i + 1 for i in powers]
# nelmts = 5

dx = 1.0 ./ nelmts
# distancefunction(x) = circle_distance_function(x,xc,rad)

err = [
    reinitialization_error(
        x -> circle_distance_function(x, xc, rad),
        ne,
        polyorder,
    ) for ne in nelmts
]
rate = convergence_rate(dx, err)
@test all(rate .> 1.95)

corner = [0.5, 0.5]
err = [
    reinitialization_error(
        x -> corner_distance_function(x, corner),
        ne,
        polyorder,
    ) for ne in nelmts
]
rate = convergence_rate(dx, err)
@test all(rate .> 1.0)
