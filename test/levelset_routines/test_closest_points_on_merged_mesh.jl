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

cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)
interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)

mergedmesh = CutCellDG.MergedMesh(cutmesh, cellquads, facequads, interfacequads)

refseedpoints, refseedcellids =
    CutCellDG.seed_zero_levelset_with_interfacequads(interfacequads, mergedmesh)
spatialseedpoints = CutCellDG.map_to_spatial(
    refseedpoints[1, :, :],
    refseedcellids[1, :],
    CutCellDG.background_mesh(mergedmesh),
)

querypoints = reshape([4.0, 1.0], 2, 1)
tol = 1e-8
boundingradius = 4.5
refclosestpoints, refclosestcellids =
    CutCellDG.closest_reference_points_on_merged_mesh(
        querypoints,
        refseedpoints,
        spatialseedpoints,
        refseedcellids,
        levelset,
        mergedmesh,
        tol,
        boundingradius,
    )

productclosestpoints = CutCellDG.map_to_spatial(
    refclosestpoints[1, :, :],
    refclosestcellids[1, :],
    CutCellDG.background_mesh(mergedmesh),
)
parentclosestpoints = CutCellDG.map_to_spatial(
    refclosestpoints[2, :, :],
    refclosestcellids[2, :],
    CutCellDG.background_mesh(mergedmesh),
)

@test allapprox(refclosestpoints[1,:,:],[-0.9,0.0])
@test allapprox(refclosestpoints[2,:,:],[1.1,0.0])
@test allapprox(productclosestpoints, [2.0 + delta, 1.0])
@test allapprox(parentclosestpoints, [2.0 + delta,1.0])
