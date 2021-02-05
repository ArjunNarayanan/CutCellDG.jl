using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("useful_routines.jl")

function displacement(alpha, x)
    u1 = alpha * x[2] * sin(pi * x[1])
    u2 = alpha * (x[1]^3 + cos(pi * x[2]))
    return [u1, u2]
end

function body_force(lambda, mu, alpha, x)
    b1 = alpha * (lambda + 2mu) * pi^2 * x[2] * sin(pi * x[1])
    b2 =
        -alpha * (6mu * x[1] + (lambda + mu) * pi * cos(pi * x[1])) +
        alpha * (lambda + 2mu) * pi^2 * cos(pi * x[2])
    return [b1, b2]
end

function onboundary(x, L, W)
    return x[2] ≈ 0.0 || x[1] ≈ L || x[2] ≈ W || x[1] ≈ 0.0
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

    basis = TensorProductBasis(2, polyorder)
    mesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
    levelset = InterpolatingPolynomial(1, basis)
    levelsetcoeffs = CutCellDG.levelset_coefficients(
        x -> plane_distance_function(x, normal, x0),
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
    CutCellDG.assemble_penalty_displacement_bc!(
        sysmatrix,
        sysrhs,
        x -> displacement(alpha, x),
        basis,
        facequads,
        stiffness,
        mergedmesh,
        x -> onboundary(x, L, W),
        penalty,
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



x0 = [0.8, 0.0]
interfaceangle = 30.0
normal = [cosd(interfaceangle), sind(interfaceangle)]
powers = [3, 4, 5]
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
u1rate = convergence_rate(dx,u1err)
u2rate = convergence_rate(dx,u2err)

@test allapprox(u1rate, repeat([2.0], length(u1rate)), 0.1)
@test allapprox(u1rate, repeat([2.0], length(u2rate)), 0.1)







x0 = [0.1, 0.0]
interfaceangle = -28.0
normal = [cosd(interfaceangle), sind(interfaceangle)]
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
u1rate = convergence_rate(dx,u1err)
u2rate = convergence_rate(dx,u2err)

@test all(u1rate .> 3.0)
@test all(u2rate .> 3.0)