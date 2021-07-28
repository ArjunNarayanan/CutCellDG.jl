using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../useful_routines.jl")

function displacement(alpha, v)
    x, y = v
    u1 = alpha * x^2 * sin(2 * pi * y)
    u2 = alpha * (x^3 + cos(2pi * y))
    return [u1, u2]
end

function body_force(lambda, mu, alpha, v)
    x, y = v

    b1 =
        alpha *
        ((lambda + 2mu) * 2 * sin(2pi * y) - mu * x^2 * 4pi^2 * sin(2pi * y))
    b2 =
        alpha * (
            2mu * x * (3 + 2pi * cos(2pi * y)) -
            (lambda + 2mu) * 4pi^2 * cos(2pi * y) +
            lambda * 2x * 2pi * cos(2pi * y)
        )
    return -[b1, b2]
end

function onboundary(x, L, W)
    return x[1] ≈ L || x[1] ≈ 0.0
end

function error_for_plane_interface(
    x0,
    normal,
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
    levelsetbasis = HermiteTensorProductBasis(2)

    dim, nf = size(interpolation_points(levelsetbasis))
    refpoints = interpolation_points(elasticitybasis)

    cgmesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], nf)
    mesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], refpoints)
    CutCellDG.make_vertical_periodic!(mesh)

    levelset = CutCellDG.LevelSet(
        x -> plane_distance_function(x, normal, x0),
        cgmesh,
        levelsetbasis,
    )

    cutmesh = CutCellDG.CutMesh(mesh, levelset)
    cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
    interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)
    facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)

    mergedmesh =
        CutCellDG.MergedMesh!(cutmesh, cellquads, facequads, interfacequads)

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
    return err
end

x0 = [0.5, 0.0]
normal = [1.0, 0.0]
powers = [3,4,5]
nelmts = [2^p + 1 for p in powers]
polyorder = 1
numqp = required_quadrature_order(polyorder)
penaltyfactor = 1e2
eta = 1

err = [
    error_for_plane_interface(
        x0,
        normal,
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

@test all(u1rate .> 1.95)
@test all(u2rate .> 1.95)


x0 = [0.5, 0.0]
normal = [1.0, 0.0]
powers = [3, 4, 5]
nelmts = [2^p + 1 for p in powers]
polyorder = 2
numqp = required_quadrature_order(polyorder)
penaltyfactor = 1e2
eta = 1

err = [
    error_for_plane_interface(
        x0,
        normal,
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

@test all(u1rate .> 2.90)
@test all(u2rate .> 2.90)
