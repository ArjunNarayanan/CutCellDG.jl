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

    elasticitybasis = LagrangeTensorProductBasis(2, polyorder)
    levelsetbasis = HermiteTensorProductBasis(2)
    quad = tensor_product_quadrature(2, 4)
    dim, nf = size(interpolation_points(levelsetbasis))
    refpoints = interpolation_points(elasticitybasis)

    cgmesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], nf)
    mesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], refpoints)

    x0 = [0.5, 0.0]
    normal = [1.0, 0.0]
    levelset = CutCellDG.LevelSet(
        x -> plane_distance_function(x, normal, x0),
        cgmesh,
        levelsetbasis,
        quad,
    )

    cutmesh = CutCellDG.CutMesh(mesh, levelset)
    cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
    interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)
    facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)

    mergedmesh = cutmesh

    onbottomboundary(x) = x[2] ≈ 0.0 ? true : false
    onrightboundary(x) = x[1] ≈ L ? true : false
    ontopboundary(x) = x[2] ≈ W ? true : false
    onleftboundary(x) = x[1] ≈ 0.0 ? true : false


    sysmatrix = CutCellDG.SystemMatrix()
    sysrhs = CutCellDG.SystemRHS()

    CutCellDG.assemble_displacement_bilinear_forms!(
        sysmatrix,
        elasticitybasis,
        cellquads,
        stiffness,
        mergedmesh,
    )
    CutCellDG.assemble_incoherent_interface_condition!(
        sysmatrix,
        elasticitybasis,
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
        elasticitybasis,
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
        elasticitybasis,
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
        elasticitybasis,
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
