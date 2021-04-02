using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../useful_routines.jl")

function test_simple_tension(polyorder)
    x0 = [0.0, 0.0]
    widths = [4.0, 1.0]
    nelements = [2, 2]

    interfacepoint = [-1.0, 0.0]
    interfacenormal = [1.0, 0.0]

    lambda, mu = 1.0, 2.0
    stiffness = CutCellDG.HookeStiffness(lambda, mu, lambda, mu)
    dx = 0.1
    e11 = dx / 4.0
    e22 = -lambda / (lambda + 2mu) * e11
    dy = e22

    penalty = 10
    eta = 1
    numqp = required_quadrature_order(polyorder)

    basis = TensorProductBasis(2, polyorder)
    cgmesh = CutCellDG.CGMesh(x0, widths, nelements, basis)
    mesh = CutCellDG.DGMesh(x0, widths, nelements, basis)

    levelset = InterpolatingPolynomial(1, basis)
    levelsetcoeffs = CutCellDG.levelset_coefficients(
        x -> plane_distance_function(x, interfacenormal, interfacepoint),
        mesh,
    )
    levelset = CutCellDG

    cutmesh = CutCellDG.CutMesh(mesh, levelset, levelsetcoeffs)

    cellquads =
        CutCellDG.CellQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
    facequads =
        CutCellDG.FaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)

    sysmatrix = CutCellDG.SystemMatrix()
    sysrhs = CutCellDG.SystemRHS()

    CutCellDG.assemble_displacement_bilinear_forms!(
        sysmatrix,
        basis,
        cellquads,
        stiffness,
        cutmesh,
    )
    CutCellDG.assemble_interelement_condition!(
        sysmatrix,
        basis,
        facequads,
        stiffness,
        cutmesh,
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
        cutmesh,
        x -> x[1] ≈ 0.0,
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
        cutmesh,
        x -> x[2] ≈ 0.0,
        [0.0, 1.0],
        penalty,
    )
    CutCellDG.assemble_penalty_displacement_component_bc!(
        sysmatrix,
        sysrhs,
        x -> dx,
        basis,
        facequads,
        stiffness,
        cutmesh,
        x -> x[1] ≈ widths[1],
        [1.0, 0.0],
        penalty,
    )

    op = CutCellDG.sparse_displacement_operator(sysmatrix, cutmesh)
    rhs = CutCellDG.displacement_rhs_vector(sysrhs, cutmesh)

    # K = Array(op)

    sol = op \ rhs
    disp = reshape(sol, 2, :)

    nodalcoordinates = CutCellDG.nodal_coordinates(cutmesh)
    testdisp = copy(nodalcoordinates)
    testdisp[1, :] .*= e11
    testdisp[2, :] .*= e22

    @test allapprox(disp, testdisp, 1e2eps())
end

test_simple_tension(1)
test_simple_tension(2)
test_simple_tension(3)
