using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../useful_routines.jl")


function test_incoherent_interface_simple_tension()
    L = 1.0
    W = 1.0
    lambda1, mu1 = 1.0, 2.0
    lambda2, mu2 = 3.0, 4.0
    penaltyfactor = 1e2
    nelmts = 1
    dx = L / nelmts
    dy = W / nelmts
    penalty = penaltyfactor / dx * (lambda1 + mu1 + lambda2 + mu2) * 0.5
    eta = 1

    applydisplacement = 0.01

    stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)
    polyorder = 1
    numqp = 2
    basis = TensorProductBasis(2, polyorder)
    mesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
    levelset = InterpolatingPolynomial(1, basis)
    x0 = [0.5, 0.0]
    normal = [1.0, 0.0]
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

    mergedmesh = cutmesh

    onbottomboundary(x) = x[2] ≈ 0.0 ? true : false
    onrightboundary(x) = x[1] ≈ L ? true : false
    ontopboundary(x) = x[2] ≈ W ? true : false
    onleftboundary(x) = x[1] ≈ 0.0 ? true : false


    sysmatrix = CutCellDG.SystemMatrix()
    sysrhs = CutCellDG.SystemRHS()

    CutCellDG.assemble_displacement_bilinear_forms!(
        sysmatrix,
        basis,
        cellquads,
        stiffness,
        mergedmesh,
    )
    CutCellDG.assemble_incoherent_interface_condition!(
        sysmatrix,
        basis,
        interfacequads,
        stiffness,
        mergedmesh,
        penalty,
        eta,
    )
    CutCellDG.assemble_penalty_displacement_component_bc!(
        sysmatrix,
        sysrhs,
        x -> 0.0,
        basis,
        facequads,
        stiffness,
        mergedmesh,
        x -> onleftboundary(x),
        [1.0, 0.0],
        penalty,
    )
    CutCellDG.assemble_penalty_displacement_component_bc!(
        sysmatrix,
        sysrhs,
        x -> 0.0,
        basis,
        facequads,
        stiffness,
        mergedmesh,
        x -> onbottomboundary(x),
        [0.0, 1.0],
        penalty,
    )
    CutCellDG.assemble_penalty_displacement_component_bc!(
        sysmatrix,
        sysrhs,
        x -> applydisplacement,
        basis,
        facequads,
        stiffness,
        mergedmesh,
        x -> onrightboundary(x),
        [1.0, 0.0],
        penalty,
    )

    matrix = CutCellDG.sparse_displacement_operator(sysmatrix, mergedmesh)
    rhs = CutCellDG.displacement_rhs_vector(sysrhs, mergedmesh)

    solution = matrix \ rhs
    nodaldisplacement = reshape(solution, 2, :)


    E1 = mu1 * (lambda1 + mu1) / (lambda1 + 2mu1)
    E2 = mu2 * (lambda2 + mu2) / (lambda2 + 2mu2)

    uI = E1 / (E1 + E2) * applydisplacement

    u10 = uI + (applydisplacement - uI) / 0.5 * (-0.5)
    Δx = applydisplacement

    u12 = -lambda1 / (lambda1 + 2mu1) * 2 * (Δx - uI)
    u22 = -lambda2 / (lambda2 + 2mu2) * 2 * uI

    testdisplacement = [
        u10 u10 Δx Δx 0.0 0.0 2uI 2uI
        0.0 u12 0.0 u12 0.0 u22 0.0 u22
    ]

    @test allapprox(nodaldisplacement, testdisplacement, 1e2eps())
end

test_incoherent_interface_simple_tension()
