using Test
using PolynomialBasis
# using Revise
using CutCellDG
include("../useful_routines.jl")

polyorder = 1
delta = 0.1
numqp = 2

L, W = 4.0, 2.0

solverbasis = LagrangeTensorProductBasis(2, polyorder)
refpoints = interpolation_points(solverbasis)
dim, nf = size(refpoints)
cgmesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], [2, 1], nf)
dgmesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [2, 1], refpoints)

normal = [1.0, 0.0]
x0 = [2.0 + delta, 0.0]
distancefunction(x) = plane_distance_function(x, normal, x0)
levelset = CutCellDG.LevelSet(distancefunction, cgmesh, solverbasis)

cutmesh = CutCellDG.CutMesh(dgmesh, levelset)

refseedpoints, refseedcellids =
    CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
spatialseedpoints =
    CutCellDG.map_to_spatial(refseedpoints, refseedcellids, cutmesh)

cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)
interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)

mergedmesh =
    CutCellDG.MergedMesh!(cutmesh, cellquads, facequads, interfacequads)
@assert CutCellDG.has_merged_cells(mergedmesh)


querypoints = reshape([4.0, 1.0], 2, 1)
tol = 1e4eps()
boundingradius = 1.5 * maximum(CutCellDG.element_size(dgmesh))
closestpoints, closestcellids, flags =
    CutCellDG.closest_points_on_zero_levelset(
        querypoints,
        spatialseedpoints,
        refseedcellids,
        levelset,
        tol,
        boundingradius,
    )

@test all(flags)

parentclosestrefpoints = CutCellDG.map_to_reference_on_merged_mesh(
    closestpoints,
    closestcellids,
    -1,
    mergedmesh,
)
productclosestrefpoints = CutCellDG.map_to_reference_on_merged_mesh(
    closestpoints,
    closestcellids,
    +1,
    mergedmesh,
)

@test allapprox(parentclosestrefpoints, [1.1, 0.0], 1e2eps())
@test allapprox(productclosestrefpoints, [-0.9, 0.0], 1e2eps())
################################################################################



################################################################################
levelsetbasis = HermiteTensorProductBasis(2)
quad = tensor_product_quadrature(2,4)
solverbasis = LagrangeTensorProductBasis(2, polyorder)
refpoints = interpolation_points(solverbasis)
dim, nf = size(interpolation_points(levelsetbasis))
cgmesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], [2, 1], nf)
dgmesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [2, 1], refpoints)

normal = [1.0, 0.0]
x0 = [2.0 + delta, 0.0]
distancefunction(x) = plane_distance_function(x, normal, x0)
levelset = CutCellDG.LevelSet(distancefunction, cgmesh, levelsetbasis, quad)

cutmesh = CutCellDG.CutMesh(dgmesh, levelset)

refseedpoints, refseedcellids =
    CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
spatialseedpoints =
    CutCellDG.map_to_spatial(refseedpoints, refseedcellids, cutmesh)

cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)
interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)

mergedmesh =
    CutCellDG.MergedMesh!(cutmesh, cellquads, facequads, interfacequads)
@assert CutCellDG.has_merged_cells(mergedmesh)


querypoints = reshape([4.0, 1.0], 2, 1)
tol = 1e4eps()
boundingradius = 1.5 * maximum(CutCellDG.element_size(dgmesh))
closestpoints, closestcellids, flags =
    CutCellDG.closest_points_on_zero_levelset(
        querypoints,
        spatialseedpoints,
        refseedcellids,
        levelset,
        tol,
        boundingradius,
    )

@test all(flags)

parentclosestrefpoints = CutCellDG.map_to_reference_on_merged_mesh(
    closestpoints,
    closestcellids,
    -1,
    mergedmesh,
)
productclosestrefpoints = CutCellDG.map_to_reference_on_merged_mesh(
    closestpoints,
    closestcellids,
    +1,
    mergedmesh,
)

@test allapprox(parentclosestrefpoints, [1.1, 0.0], 1e3eps())
@test allapprox(productclosestrefpoints, [-0.9, 0.0], 1e3eps())
