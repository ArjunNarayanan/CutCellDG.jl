using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../useful_routines.jl")
include("test_problem_and_solver.jl")

function error_for_curved_interface(
    xc,
    radius,
    nelmts,
    polyorder,
    numqp,
    penaltyfactor,
    eta,
)
    L = 1.0
    W = 1.0
    lambda, mu = 1.0, 2.0
    dx = 1.0 / nelmts
    penalty = penaltyfactor / dx * (lambda + mu)
    alpha = 0.1
    stiffness = CutCellDG.HookeStiffness(lambda, mu, lambda, mu)

    elasticitybasis = LagrangeTensorProductBasis(2, polyorder)

    mergedmesh, cellquads, facequads, interfacequads, levelset =
        construct_mesh_and_quadratures(
            [L, W],
            nelmts,
            elasticitybasis,
            x -> circle_distance_function(x, xc, radius)[1],
            numqp,
        )

    sysmatrix = CutCellDG.SystemMatrix()
    sysrhs = CutCellDG.SystemRHS()

    CutCellDG.assemble_displacement_bilinear_forms!(
        sysmatrix,
        elasticitybasis,
        cellquads,
        stiffness,
        mergedmesh,
    )
    CutCellDG.assemble_interelement_condition!(
        sysmatrix,
        elasticitybasis,
        facequads,
        stiffness,
        mergedmesh,
        penalty,
        eta,
    )
    CutCellDG.assemble_coherent_interface_condition!(
        sysmatrix,
        elasticitybasis,
        interfacequads,
        stiffness,
        mergedmesh,
        penalty,
        eta,
    )
    CutCellDG.assemble_penalty_displacement_bc!(
        sysmatrix,
        sysrhs,
        x -> displacement(alpha, x),
        elasticitybasis,
        facequads,
        stiffness,
        mergedmesh,
        x -> onboundary(x, L, W),
        penalty,
    )
    CutCellDG.assemble_body_force!(
        sysrhs,
        x -> body_force(lambda, mu, alpha, x),
        elasticitybasis,
        cellquads,
        mergedmesh,
    )

    matrix = CutCellDG.sparse_displacement_operator(sysmatrix, mergedmesh)
    rhs = CutCellDG.displacement_rhs_vector(sysrhs, mergedmesh)

    solution = matrix \ rhs
    nodaldisplacement = reshape(solution, 2, :)

    err = mesh_L2_error(
        nodaldisplacement,
        x -> displacement(alpha, x),
        elasticitybasis,
        cellquads,
        mergedmesh,
    )
end

function test_edge_intersecting_curved_interface()
    xc = [1.0, 0.5]
    radius = 0.45
    polyorder = 2
    penaltyfactor = 1e2
    powers = [3, 4, 5]
    nelmts = [2^p + 1 for p in powers]
    numqp = required_quadrature_order(polyorder) + 2
    nelmts = [2^p + 1 for p in powers]
    eta = 1

    err = [
        error_for_curved_interface(
            xc,
            radius,
            ne,
            polyorder,
            numqp,
            penaltyfactor,
            eta,
        ) for ne in nelmts
    ]
    u1err = [er[1] for er in err]
    u2err = [er[2] for er in err]
    dx = 1.0 ./ nelmts

    u1rate = diff(log.(u1err)) ./ diff(log.(dx))
    u2rate = diff(log.(u2err)) ./ diff(log.(dx))

    @test allapprox(u1rate, repeat([3.0], length(u1rate)), 0.05)
    @test allapprox(u2rate, repeat([3.0], length(u2rate)), 0.05)
end




function test_circular_interface()
    xc = [0.3, 0.7]
    radius = 0.2
    polyorder = 2
    penaltyfactor = 1e2
    powers = [3, 4, 5]
    nelmts = [2^p + 1 for p in powers]
    numqp = required_quadrature_order(polyorder) + 2
    nelmts = [2^p + 1 for p in powers]
    eta = 1

    err = [
        error_for_curved_interface(
            xc,
            radius,
            ne,
            polyorder,
            numqp,
            penaltyfactor,
            eta,
        ) for ne in nelmts
    ]
    u1err = [er[1] for er in err]
    u2err = [er[2] for er in err]
    dx = 1.0 ./ nelmts

    u1rate = diff(log.(u1err)) ./ diff(log.(dx))
    u2rate = diff(log.(u2err)) ./ diff(log.(dx))

    @test all(u1rate .> 2.95)
    @test all(u2rate .> 2.95)
end

test_edge_intersecting_curved_interface()
test_circular_interface()
