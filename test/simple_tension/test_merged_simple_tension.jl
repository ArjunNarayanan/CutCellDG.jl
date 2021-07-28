using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../useful_routines.jl")

function exact_displacement(x, e11, e22)
    return [x[1] * e11, x[2] * e22]
end

function test_merged_simple_tension()
    polyorder = 1
    numqp = 2

    widths = [2.0, 1.0]
    nelmts = [5, 5]

    elasticitybasis = LagrangeTensorProductBasis(2, polyorder)
    levelsetbasis = HermiteTensorProductBasis(2)

    dim, nf = size(interpolation_points(levelsetbasis))
    refpoints = interpolation_points(elasticitybasis)

    cgmesh = CutCellDG.CGMesh([0.0, 0.0], widths, nelmts, nf)
    mesh = CutCellDG.DGMesh([0.0, 0.0], widths, nelmts, refpoints)

    interfaceangle = 40.0
    normal = [cosd(interfaceangle), sind(interfaceangle)]
    x0 = [1.8, 0.0]
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
    @assert CutCellDG.has_merged_cells(mergedmesh)

    lambda, mu = (1.0, 2.0)
    penalty = 1.0
    eta = 1
    dx = 0.1
    e11 = dx / 2.0
    e22 = -lambda / (lambda + 2mu) * e11
    dy = e22
    s11 = (lambda + 2mu) * e11 + lambda * e22

    stiffness = CutCellDG.HookeStiffness(lambda, mu, lambda, mu)

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
    CutCellDG.assemble_penalty_displacement_component_bc!(
        sysmatrix,
        sysrhs,
        x -> 0.0,
        elasticitybasis,
        facequads,
        stiffness,
        mergedmesh,
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
        mergedmesh,
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
        mergedmesh,
        x -> x[1] ≈ widths[1],
        [1.0, 0.0],
        penalty,
    )


    matrix = CutCellDG.sparse_displacement_operator(sysmatrix, mergedmesh)
    rhs = CutCellDG.displacement_rhs_vector(sysrhs, mergedmesh)

    solution = matrix \ rhs
    displacement = reshape(solution, 2, :)

    err = mesh_L2_error(
        displacement,
        x -> exact_displacement(x, e11, e22),
        elasticitybasis,
        cellquads,
        mergedmesh,
    )

    @test allapprox(err, zeros(2), 1e2eps())
end

test_merged_simple_tension()
