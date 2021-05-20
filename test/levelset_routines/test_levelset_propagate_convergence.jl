using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../../test/useful_routines.jl")

function time_step_size(levelsetspeed, dx; CFL = 0.5)
    s = maximum(abs.(levelsetspeed))
    return CFL * dx / s
end

function step_interface(
    levelset,
    levelsetspeed,
    dt,
    paddedmesh,
    tol,
    boundingradius,
)
    mesh = CutCellDG.background_mesh(levelset)
    cutmesh = CutCellDG.CutMesh(mesh, levelset)


    refseedpoints, refseedcellids =
        CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
    spatialseedpoints =
        CutCellDG.map_to_spatial(refseedpoints, refseedcellids, mesh)

    paddedlevelset = CutCellDG.BoundaryPaddedLevelSet(
        paddedmesh,
        spatialseedpoints,
        refseedcellids,
        levelset,
        tol,
        boundingradius,
    )

    return CutCellDG.step_first_order_levelset(
        paddedlevelset,
        levelsetspeed,
        dt,
    )
end

function signed_distance_coefficients(levelset, mesh, tol, boundingradius)
    cutmesh = CutCellDG.CutMesh(mesh, levelset)
    refseedpoints, refseedcellids =
        CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
    spatialseedpoints =
        CutCellDG.map_to_spatial(refseedpoints, refseedcellids, mesh)
    nodalcoordinates = CutCellDG.nodal_coordinates(mesh)

    signeddistance = CutCellDG.distance_to_zero_levelset(
        nodalcoordinates,
        spatialseedpoints,
        refseedcellids,
        levelset,
        tol,
        boundingradius,
    )

    return signeddistance
end

function run_time_steps(levelset, levelsetspeed, paddedmesh, dt, numiter)
    mesh = CutCellDG.background_mesh(levelset)
    dx = CutCellDG.element_size(mesh)
    polyorder = CutCellDG.order(levelset)
    tol = 1e-2(minimum(dx))^(polyorder + 1)
    boundingradius = 1.5maximum(dx)

    numnodes = CutCellDG.number_of_nodes(mesh)
    newcoeffs = zeros(numnodes)
    itercoeffs = zeros(numnodes, numiter + 1)
    itercoeffs[:, 1] = CutCellDG.coefficients(levelset)

    for iter = 1:numiter
        newcoeffs = step_interface(
            levelset,
            levelsetspeed,
            dt,
            paddedmesh,
            tol,
            boundingradius,
        )
        CutCellDG.update_coefficients!(levelset, newcoeffs)
        dist = signed_distance_coefficients(levelset, mesh, tol, boundingradius)
        CutCellDG.update_coefficients!(levelset, dist)
        newcoeffs = dist
    end
    return newcoeffs
end

function grid_range(mesh)
    x0 = CutCellDG.reference_corner(mesh)
    w = CutCellDG.mesh_widths(mesh)
    nfmside = CutCellDG.nodes_per_mesh_side(mesh)

    x = range(x0[1], stop = x0[1] + w[1], length = nfmside[1])
    y = range(x0[2], stop = x0[2] + w[2], length = nfmside[2])

    return x, y
end

function error_for_nelmts(nelmts)
    x0 = [0.0, 0.0]
    L, W = 1.0, 1.0
    polyorder = 2
    numqp = required_quadrature_order(polyorder)

    xc = [0.5, 0.5]
    radius = 0.45
    speed = -1.0
    stoptime = 0.25

    basis = LagrangeTensorProductBasis(2, polyorder)
    dim,nf = size(interpolation_points(basis))
    quad = tensor_product_quadrature(2, numqp)
    mesh = CutCellDG.CGMesh(x0, [L, W], [nelmts, nelmts], nf)
    paddedmesh = CutCellDG.BoundaryPaddedMesh(mesh, 1)
    numnodes = CutCellDG.number_of_nodes(mesh)
    dx = minimum(CutCellDG.grid_size(paddedmesh))

    levelset = CutCellDG.LevelSet(
        x -> -circle_distance_function(x, xc, radius),
        mesh,
        basis,
    )
    levelsetspeed = speed * ones(numnodes)
    dt = time_step_size(levelsetspeed, dx)
    @assert isinteger(stoptime / dt)
    numiter = round(Int, stoptime / dt)
    itercoeffs =
        run_time_steps(levelset, levelsetspeed, paddedmesh, dt, numiter)

    finalradius = radius - stoptime
    err = uniform_mesh_L2_error(
        itercoeffs',
        x -> -circle_distance_function(x, xc, finalradius)[1],
        basis,
        quad,
        mesh,
    )[1]

    normalizer = integral_norm_on_uniform_mesh(
        x -> -circle_distance_function(x, xc, finalradius)[1],
        quad,
        mesh,
        1,
    )[1]
    normalizederr = err / normalizer

    return normalizederr
end

powers = 2:5
nelmts = 2 .^powers
err = error_for_nelmts.(nelmts)
dx = 1.0 ./ nelmts
rate = convergence_rate(dx,err)

@test all(rate .> 0.9)
