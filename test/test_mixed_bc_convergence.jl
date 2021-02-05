using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("useful_routines.jl")

function displacement(alpha, x)
    u1 = alpha * x[2] * sin(pi * x[1])
    u2 = alpha * (x[1]^3 + cos(pi * x[2]))
    return [u1, u2]
end

function stress_field(lambda, mu, alpha, x)
    s11 =
        (lambda + 2mu) * alpha * pi * x[2] * cos(pi * x[1]) -
        lambda * alpha * pi * sin(pi * x[2])
    s22 =
        -(lambda + 2mu) * alpha * pi * sin(pi * x[2]) +
        lambda * alpha * pi * x[2] * cos(pi * x[1])
    s12 = alpha * mu * (3x[1]^2 + sin(pi * x[1]))
    return [s11, s22, s12]
end

function body_force(lambda, mu, alpha, x)
    b1 = alpha * (lambda + 2mu) * pi^2 * x[2] * sin(pi * x[1])
    b2 =
        -alpha * (6mu * x[1] + (lambda + mu) * pi * cos(pi * x[1])) +
        alpha * (lambda + 2mu) * pi^2 * cos(pi * x[2])
    return [b1, b2]
end

function onleftboundary(x, L, W)
    return x[1] ≈ 0.0
end

function onbottomboundary(x, L, W)
    return x[2] ≈ 0.0
end

function onrightboundary(x, L, W)
    return x[1] ≈ L
end

function ontopboundary(x, L, W)
    return x[2] ≈ W
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
    mesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
    levelset = InterpolatingPolynomial(1, basis)
    levelsetcoeffs = CutCellDG.levelset_coefficients(
        x -> circle_distance_function(x, xc, radius),
        mesh,
    )

    cutmesh = CutCellDG.CutMesh(mesh, levelset, levelsetcoeffs)
    cellquads =
        CutCellDG.CellQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
    interfacequads =
        CutCellDG.InterfaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
    facequads =
        CutCellDG.FaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)

    mergedwithcell, hasmergedcells = CutCellDG.merge_tiny_cells_in_mesh!(
        cutmesh,
        cellquads,
        facequads,
        interfacequads,
    )
    @assert hasmergedcells
    mergedmesh = CutCellDG.MergedMesh(cutmesh, mergedwithcell)

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
    CutCellDG.assemble_body_force!(
        sysrhs,
        x -> body_force(lambda, mu, alpha, x),
        basis,
        cellquads,
        mergedmesh,
    )

    CutCellDG.assemble_penalty_displacement_component_bc!(
        sysmatrix,
        sysrhs,
        x -> displacement(alpha, x)[2],
        basis,
        facequads,
        stiffness,
        mergedmesh,
        x -> onbottomboundary(x, L, W),
        [0.0, 1.0],
        penalty,
    )
    CutCellDG.assemble_traction_force_component_linear_form!(
        sysrhs,
        x -> -stress_field(lambda, mu, alpha, x)[3],
        basis,
        facequads,
        mergedmesh,
        x -> onbottomboundary(x, L, W),
        [1.0, 0.0],
    )

    CutCellDG.assemble_penalty_displacement_component_bc!(
        sysmatrix,
        sysrhs,
        x -> displacement(alpha, x)[1],
        basis,
        facequads,
        stiffness,
        mergedmesh,
        x -> onleftboundary(x, L, W),
        [1.0, 0.0],
        penalty,
    )
    CutCellDG.assemble_traction_force_component_linear_form!(
        sysrhs,
        x -> -stress_field(lambda, mu, alpha, x)[3],
        basis,
        facequads,
        mergedmesh,
        x -> onleftboundary(x, L, W),
        [0.0, 1.0],
    )

    CutCellDG.assemble_traction_force_linear_form!(
        sysrhs,
        x -> stress_field(lambda, mu, alpha, x)[[1, 3]],
        basis,
        facequads,
        mergedmesh,
        x -> onrightboundary(x, L, W),
    )
    CutCellDG.assemble_traction_force_linear_form!(
        sysrhs,
        x -> stress_field(lambda, mu, alpha, x)[[3, 2]],
        basis,
        facequads,
        mergedmesh,
        x -> ontopboundary(x, L, W),
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




xc = [0.3, 0.7]
radius = 0.15
polyorder = 3
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

@test all(u1rate .> 3.95)
@test all(u2rate .> 3.95)
