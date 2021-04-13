using Test
using PolynomialBasis
using Revise
using CutCellDG
include("../useful_routines.jl")


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

cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)
interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)

mergedmesh = CutCellDG.MergedMesh(cutmesh, cellquads, facequads, interfacequads)

refseedpoints, refseedcellids =
    CutCellDG.seed_zero_levelset_with_interfacequads(interfacequads, mergedmesh)
parentspatialseedpoints = CutCellDG.map_to_spatial(
    refseedpoints[1, :, :],
    refseedcellids[1, :],
    CutCellDG.background_mesh(mergedmesh),
)
productspatialseedpoints = CutCellDG.map_to_spatial(
    refseedpoints[2, :, :],
    refseedcellids[2, :],
    CutCellDG.background_mesh(mergedmesh),
)

@test allapprox(parentspatialseedpoints, productspatialseedpoints, 1e2eps())

querypoints = CutCellDG.nodal_coordinates(cgmesh)

tol = 1e4eps()
boundingradius = 4.5
refclosestpoints, refclosestcellids =
    CutCellDG.closest_reference_points_on_merged_mesh(
        querypoints,
        refseedpoints,
        parentspatialseedpoints,
        refseedcellids,
        levelset,
        mergedmesh,
        tol,
        boundingradius,
    )

parentclosestpoints = CutCellDG.map_to_spatial(
    refclosestpoints[1, :, :],
    refclosestcellids[1, :],
    CutCellDG.background_mesh(mergedmesh),
)
productclosestpoints = CutCellDG.map_to_spatial(
    refclosestpoints[2, :, :],
    refclosestcellids[2, :],
    CutCellDG.background_mesh(mergedmesh),
)

@test allapprox(parentclosestpoints, productclosestpoints, 1e-2)
testclosestpoints = closest_point_on_arc(querypoints, center, radius)
@test allapprox(parentclosestpoints[:,2:end],testclosestpoints[:,2:end],1e-2)
