using Test
using PolynomialBasis
# using Revise
using CutCellDG
include("../useful_routines.jl")

polyorder = 2
delta = 0.1
numqp = 2
L, W = 4.0, 4.0
basis = TensorProductBasis(2, polyorder)
cgmesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], [2, 2], basis)
dgmesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [2, 2], basis)

center = [0.0, 0.0]
radius = 2 * sqrt(2) + delta
distancefunction(x) = circle_distance_function(x, center, radius)
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

using LinearAlgebra
distance = vec(mapslices(norm,parentclosestpoints,dims=1))

@test allapprox(distance,radius*ones(length(distance)),0.05)


################################################################################


polyorder = 2
numqp = 2
L, W = 1.0, 1.0
nelmts = 17
basis = TensorProductBasis(2, polyorder)
cgmesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
dgmesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)

center = [0.0, 0.0]
radius = 0.5
distancefunction(x) = circle_distance_function(x, center, radius)
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

using LinearAlgebra
distance = vec(mapslices(norm,parentclosestpoints,dims=1))

@test allapprox(distance,radius*ones(length(distance)),1e-4)
