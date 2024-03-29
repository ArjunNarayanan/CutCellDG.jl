using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../useful_routines.jl")

function test_cut_simple_tension()
    x0 = [0.0, 0.0]
    widths = [4.0, 1.0]
    nelements = [2, 2]

    interfacepoint = [1.0, 0.0]
    interfaceangle = 10.0
    interfacenormal = [cosd(interfaceangle), sind(interfaceangle)]

    lambda, mu = 1.0, 2.0
    stiffness = CutCellDG.HookeStiffness(lambda, mu, lambda, mu)
    penalty = 10.0
    eta = 1
    dx = 0.1
    e11 = dx / 4.0
    e22 = -lambda / (lambda + 2mu) * e11
    dy = e22

    polyorder = 2
    numqp = required_quadrature_order(polyorder)

    elasticitybasis = LagrangeTensorProductBasis(2, polyorder)
    levelsetbasis = HermiteTensorProductBasis(2)
    # quad = tensor_product_quadrature(2,4)
    dim,nf = size(interpolation_points(levelsetbasis))
    refpoints = interpolation_points(elasticitybasis)

    cgmesh = CutCellDG.CGMesh(x0, widths, nelements, nf)
    mesh = CutCellDG.DGMesh(x0, widths, nelements, refpoints)

    levelset = CutCellDG.LevelSet(
        x -> plane_distance_function(x, interfacenormal, interfacepoint),
        cgmesh,
        levelsetbasis,
    )

    cutmesh = CutCellDG.CutMesh(mesh, levelset)

    cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
    facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)
    interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)

    sysmatrix = CutCellDG.SystemMatrix()
    sysrhs = CutCellDG.SystemRHS()

    CutCellDG.assemble_displacement_bilinear_forms!(
        sysmatrix,
        elasticitybasis,
        cellquads,
        stiffness,
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
    CutCellDG.assemble_penalty_displacement_component_bc!(
        sysmatrix,
        sysrhs,
        x -> dx,
        elasticitybasis,
        facequads,
        stiffness,
        cutmesh,
        x -> x[1] ≈ widths[1],
        [1.0, 0.0],
        penalty,
    )
    CutCellDG.assemble_coherent_interface_condition!(
        sysmatrix,
        elasticitybasis,
        interfacequads,
        stiffness,
        cutmesh,
        penalty,
        eta,
    )

    op = CutCellDG.sparse_displacement_operator(sysmatrix, cutmesh)
    rhs = CutCellDG.displacement_rhs_vector(sysrhs, cutmesh)
    #
    sol = op \ rhs
    disp = reshape(sol, 2, :)

    nodalcoordinates = CutCellDG.nodal_coordinates(cutmesh)
    nodalcoordinates = hcat(nodalcoordinates, nodalcoordinates[:, 1:18])

    testdisp = copy(nodalcoordinates)
    testdisp[1, :] .*= e11
    testdisp[2, :] .*= e22

    @test allapprox(disp, testdisp, 1e3eps())
end

test_cut_simple_tension()
