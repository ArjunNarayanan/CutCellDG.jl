using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../useful_routines.jl")
include("test_problem_and_solver.jl")

function ondisplacementboundary(x, L, W)
    return onbottomboundary(x, L, W) ||
           ontopboundary(x, L, W) ||
           onleftboundary(x, L, W)
end

function ontractionboundary(x, L, W)
    return onrightboundary(x, L, W)
end

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

    basis = TensorProductBasis(2, polyorder)

    mergedmesh, cellquads, facequads, interfacequads =
        construct_mesh_and_quadratures(
            [L, W],
            nelmts,
            basis,
            x -> circle_distance_function(x, xc, radius),
            numqp,
        )

    sysmatrix = CutCellDG.SystemMatrix()
    sysrhs = CutCellDG.SystemRHS()

    CutCellDG.assemble_displacement_bilinear_forms!(
        sysmatrix,
        basis,
        cellquads,
        stiffness,
        mergedmesh,
    )
    CutCellDG.assemble_interelement_condition!(
        sysmatrix,
        basis,
        facequads,
        stiffness,
        mergedmesh,
        penalty,
        eta,
    )
    CutCellDG.assemble_coherent_interface_condition!(
        sysmatrix,
        basis,
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
        basis,
        facequads,
        stiffness,
        mergedmesh,
        x -> ondisplacementboundary(x, L, W),
        penalty,
    )
    CutCellDG.assemble_traction_force_linear_form!(
        sysrhs,
        x -> stress_field(lambda, mu, alpha, x)[[1, 3]],
        basis,
        facequads,
        mergedmesh,
        x -> ontractionboundary(x, L, W),
    )
    CutCellDG.assemble_body_force!(
        sysrhs,
        x -> body_force(lambda, mu, alpha, x),
        basis,
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
        basis,
        cellquads,
        mergedmesh,
    )
end


function test_displacement_and_traction_bc_edge_intersecting_curved_interface()
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

    @test all(u1rate .> 2.95)
    @test all(u2rate .> 2.95)
end




function test_displacement_and_traction_bc_circular_interface()
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

    u1rate = convergence_rate(dx, u1err)
    u2rate = convergence_rate(dx, u2err)

    @test all(u1rate .> 2.95)
    @test all(u2rate .> 2.95)
end



test_displacement_and_traction_bc_edge_intersecting_curved_interface()
test_displacement_and_traction_bc_circular_interface()
