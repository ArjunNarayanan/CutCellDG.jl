using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../useful_routines.jl")


function test_transformation_strain_simple_tension()
    L = 1.0
    W = 1.0
    lambda1, mu1 = 1.0, 2.0
    K1 = (lambda1 + 2mu1 / 3)
    lambda2, mu2 = 2.0, 4.0
    theta0 = -0.067
    e22 = K1 * theta0 / (lambda1 + 2mu1)
    penaltyfactor = 1e2
    nelmts = 2
    dx = 1.0 / nelmts
    penalty = penaltyfactor / dx * (lambda1 + mu1)
    eta = 1
    stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)
    transfstress =
        CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)


    polyorder = 1
    numqp = required_quadrature_order(polyorder) + 2

    elasticitybasis = LagrangeTensorProductBasis(2, polyorder)
    levelsetbasis = HermiteTensorProductBasis(2)
    quad = tensor_product_quadrature(2, 4)
    dim, nf = size(interpolation_points(levelsetbasis))
    refpoints = interpolation_points(elasticitybasis)

    cgmesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], nf)
    mesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], refpoints)

    levelset =
        CutCellDG.LevelSet(x -> 1.0, cgmesh, levelsetbasis, quad)

    cutmesh = CutCellDG.CutMesh(mesh, levelset)
    cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
    interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)
    facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)

    sysmatrix = CutCellDG.SystemMatrix()
    sysrhs = CutCellDG.SystemRHS()

    CutCellDG.assemble_displacement_bilinear_forms!(
        sysmatrix,
        elasticitybasis,
        cellquads,
        stiffness,
        cutmesh,
    )
    CutCellDG.assemble_bulk_transformation_linear_form!(
        sysrhs,
        transfstress,
        elasticitybasis,
        cellquads,
        cutmesh,
    )
    CutCellDG.assemble_interelement_condition!(
        sysmatrix,
        elasticitybasis,
        facequads,
        stiffness,
        cutmesh,
        penalty,
        eta,
    )
    CutCellDG.assemble_interelement_transformation_linear_form!(
        sysrhs,
        transfstress,
        elasticitybasis,
        facequads,
        cutmesh,
    )

    CutCellDG.assemble_penalty_displacement_component_bc!(
        sysmatrix,
        sysrhs,
        x -> 0.0,
        elasticitybasis,
        facequads,
        stiffness,
        cutmesh,
        x -> x[1] ≈ 0.0,
        [1.0, 0.0],
        penalty,
    )
    CutCellDG.assemble_penalty_displacement_component_transformation_linear_form!(
        sysrhs,
        transfstress,
        elasticitybasis,
        facequads,
        cutmesh,
        x -> x[1] ≈ 0.0,
        [1.0, 0.0],
    )

    CutCellDG.assemble_penalty_displacement_component_bc!(
        sysmatrix,
        sysrhs,
        x -> 0.0,
        elasticitybasis,
        facequads,
        stiffness,
        cutmesh,
        x -> x[2] ≈ 0.0,
        [0.0, 1.0],
        penalty,
    )
    CutCellDG.assemble_penalty_displacement_component_transformation_linear_form!(
        sysrhs,
        transfstress,
        elasticitybasis,
        facequads,
        cutmesh,
        x -> x[2] ≈ 0.0,
        [0.0, 1.0],
    )

    CutCellDG.assemble_penalty_displacement_component_bc!(
        sysmatrix,
        sysrhs,
        x -> 0.0,
        elasticitybasis,
        facequads,
        stiffness,
        cutmesh,
        x -> x[1] ≈ L,
        [1.0, 0.0],
        penalty,
    )
    CutCellDG.assemble_penalty_displacement_component_transformation_linear_form!(
        sysrhs,
        transfstress,
        elasticitybasis,
        facequads,
        cutmesh,
        x -> x[1] ≈ L,
        [1.0, 0.0],
    )


    matrix = CutCellDG.sparse_displacement_operator(sysmatrix, cutmesh)
    rhs = CutCellDG.displacement_rhs_vector(sysrhs, cutmesh)

    sol = matrix \ rhs
    disp = reshape(sol, 2, :)

    nodalcoordinates = CutCellDG.nodal_coordinates(mesh)

    testdisp = copy(nodalcoordinates)
    testdisp[1, :] .= 0.0
    testdisp[2, :] .*= e22

    @test allapprox(disp, testdisp, 1e2eps())
end

test_transformation_strain_simple_tension()
