using Test
using PolynomialBasis
# using Revise
using CutCellDG
include("../useful_routines.jl")

polyorder = 1
delta = 0.1
numqp = 2

L, W = 4.0, 2.0
basis = TensorProductBasis(2, polyorder)
cgmesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], [2, 1], basis)
dgmesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [2, 1], basis)

normal = [1.0, 0.0]
x0 = [2.0 + delta, 0.0]
distancefunction(x) = plane_distance_function(x, normal, x0)
levelset = CutCellDG.LevelSet(distancefunction, cgmesh, basis)

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


querypoints = reshape([4.0, 1.0], 2, 1)
tol = 1e4eps()
boundingradius = 4.5
maxiter = 20
refclosestpoints, refclosestcellids =
    CutCellDG.closest_reference_points_on_levelset(
        querypoints,
        refseedpoints,
        spatialseedpoints,
        refseedcellids,
        levelset,
        tol,
        boundingradius,
        maxiter
    )

parentclosestrefpoints = CutCellDG.map_reference_points_to_merged_mesh(
    refclosestpoints,
    refclosestcellids,
    -1,
    mergedmesh,
)
productclosestrefpoints = CutCellDG.map_reference_points_to_merged_mesh(
    refclosestpoints,
    refclosestcellids,
    +1,
    mergedmesh,
)

parentclosestpoints = CutCellDG.map_to_spatial_on_merged_mesh(
    parentclosestrefpoints,
    refclosestcellids,
    -1,
    mergedmesh,
)
productclosestpoints = CutCellDG.map_to_spatial_on_merged_mesh(
    productclosestrefpoints,
    refclosestcellids,
    +1,
    mergedmesh,
)


@test allapprox(parentclosestrefpoints, [1.1, 0.0], 1e2eps())
@test allapprox(productclosestrefpoints, [-0.9, 0.0], 1e2eps())
@test allapprox(productclosestpoints, [2.0 + delta, 1.0], 1e2eps())
@test allapprox(parentclosestpoints, [2.0 + delta, 1.0], 1e2eps())

################################################################################

polyorder = 1
delta = 0.1
numqp = 2
L, W = 4.0, 4.0
basis = TensorProductBasis(2, polyorder)
cgmesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], [2, 2], basis)
dgmesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [2, 2], basis)

normal = [1.0, 1.0] / sqrt(2)
x0 = [2.0, 2.0 + delta]
distancefunction(x) = plane_distance_function(x, normal, x0)
levelset = CutCellDG.LevelSet(distancefunction, cgmesh, basis)

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
@test CutCellDG.has_merged_cells(mergedmesh)

querypoints = CutCellDG.nodal_coordinates(cgmesh)

tol = 1e4eps()
boundingradius = 4.5
maxiter = 20
refclosestpoints, refclosestcellids =
    CutCellDG.closest_reference_points_on_levelset(
        querypoints,
        refseedpoints,
        spatialseedpoints,
        refseedcellids,
        levelset,
        tol,
        boundingradius,
        maxiter
    )

parentclosestrefpoints = CutCellDG.map_reference_points_to_merged_mesh(
    refclosestpoints,
    refclosestcellids,
    -1,
    mergedmesh,
)
productclosestrefpoints = CutCellDG.map_reference_points_to_merged_mesh(
    refclosestpoints,
    refclosestcellids,
    +1,
    mergedmesh,
)

parentclosestpoints = CutCellDG.map_to_spatial_on_merged_mesh(
    parentclosestrefpoints,
    refclosestcellids,
    -1,
    mergedmesh,
)
productclosestpoints = CutCellDG.map_to_spatial_on_merged_mesh(
    productclosestrefpoints,
    refclosestcellids,
    +1,
    mergedmesh,
)


@test allapprox(parentclosestpoints, productclosestpoints, 1e2eps())

testcp = closest_point_on_plane(querypoints, normal, x0)
@test allapprox(parentclosestpoints, testcp, 1e2eps())


################################################################################

polyorder = 1
delta = 0.05
numqp = 2
L, W = 2.0, 1.0
basis = TensorProductBasis(2, polyorder)
cgmesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], [2, 2], basis)
dgmesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [2, 2], basis)

normal = [1.0, 1.0] / sqrt(2)
x0 = [1.0, 0.5 + delta]
distancefunction(x) = plane_distance_function(x, normal, x0)
levelset = CutCellDG.LevelSet(distancefunction, cgmesh, basis)

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
@test CutCellDG.has_merged_cells(mergedmesh)

querypoints = CutCellDG.nodal_coordinates(cgmesh)

tol = 1e4eps()
boundingradius = 4.0
maxiter = 20
refclosestpoints, refclosestcellids =
    CutCellDG.closest_reference_points_on_levelset(
        querypoints,
        refseedpoints,
        spatialseedpoints,
        refseedcellids,
        levelset,
        tol,
        boundingradius,
        maxiter
    )

parentclosestrefpoints = CutCellDG.map_reference_points_to_merged_mesh(
    refclosestpoints,
    refclosestcellids,
    -1,
    mergedmesh,
)
productclosestrefpoints = CutCellDG.map_reference_points_to_merged_mesh(
    refclosestpoints,
    refclosestcellids,
    +1,
    mergedmesh,
)

parentclosestpoints = CutCellDG.map_to_spatial_on_merged_mesh(
    parentclosestrefpoints,
    refclosestcellids,
    -1,
    mergedmesh,
)
productclosestpoints = CutCellDG.map_to_spatial_on_merged_mesh(
    productclosestrefpoints,
    refclosestcellids,
    +1,
    mergedmesh,
)

@test allapprox(parentclosestpoints, productclosestpoints, 1e2eps())

testcp = closest_point_on_plane(querypoints, normal, x0)
@test allapprox(parentclosestpoints, testcp, 1e2eps())


################################################################################
