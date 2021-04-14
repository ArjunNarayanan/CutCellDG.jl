using Test
using PolynomialBasis
using Revise
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



# cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
# facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)
# interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)
#
# mergedmesh = CutCellDG.MergedMesh(cutmesh, cellquads, facequads, interfacequads)
#
# refseedpoints, refseedcellids =
#     CutCellDG.seed_zero_levelset_with_interfacequads(interfacequads, mergedmesh)
# spatialseedpoints = CutCellDG.map_to_spatial(
#     refseedpoints[1, :, :],
#     refseedcellids[1, :],
#     CutCellDG.background_mesh(mergedmesh),
# )
#
# querypoints = reshape([4.0, 1.0], 2, 1)
# tol = 1e-8
# boundingradius = 4.5
# refclosestpoints, refclosestcellids =
#     CutCellDG.closest_reference_points_on_merged_mesh(
#         querypoints,
#         refseedpoints,
#         spatialseedpoints,
#         refseedcellids,
#         levelset,
#         mergedmesh,
#         tol,
#         boundingradius,
#     )
#
# productclosestpoints = CutCellDG.map_to_spatial(
#     refclosestpoints[1, :, :],
#     refclosestcellids[1, :],
#     CutCellDG.background_mesh(mergedmesh),
# )
# parentclosestpoints = CutCellDG.map_to_spatial(
#     refclosestpoints[2, :, :],
#     refclosestcellids[2, :],
#     CutCellDG.background_mesh(mergedmesh),
# )
#
# @test allapprox(refclosestpoints[1, :, :], [-0.9, 0.0], 1e2eps())
# @test allapprox(refclosestpoints[2, :, :], [1.1, 0.0], 1e2eps())
# @test allapprox(productclosestpoints, [2.0 + delta, 1.0], 1e2eps())
# @test allapprox(parentclosestpoints, [2.0 + delta, 1.0], 1e2eps())
#
# ################################################################################
#
# polyorder = 1
# delta = 0.1
# numqp = 2
# L, W = 4.0, 4.0
# basis = TensorProductBasis(2, polyorder)
# cgmesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], [2, 2], basis)
# dgmesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [2, 2], basis)
#
# normal = [1.0, 1.0] / sqrt(2)
# x0 = [2.0, 2.0 + delta]
# distancefunction(x) = plane_distance_function(x, normal, x0)
# levelset = CutCellDG.LevelSet(distancefunction, cgmesh, basis)
#
# cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
#
# cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
# facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)
# interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)
#
# mergedmesh = CutCellDG.MergedMesh(cutmesh, cellquads, facequads, interfacequads)
#
# refseedpoints, refseedcellids =
#     CutCellDG.seed_zero_levelset_with_interfacequads(interfacequads, mergedmesh)
# parentspatialseedpoints = CutCellDG.map_to_spatial(
#     refseedpoints[1, :, :],
#     refseedcellids[1, :],
#     CutCellDG.background_mesh(mergedmesh),
# )
# productspatialseedpoints = CutCellDG.map_to_spatial(
#     refseedpoints[2, :, :],
#     refseedcellids[2, :],
#     CutCellDG.background_mesh(mergedmesh),
# )
#
# @test allapprox(parentspatialseedpoints, productspatialseedpoints, 1e2eps())
#
# querypoints = CutCellDG.nodal_coordinates(cgmesh)
#
# tol = 1e-14
# boundingradius = 1.0
# refclosestpoints, refclosestcellids =
#     CutCellDG.closest_reference_points_on_merged_mesh(
#         querypoints,
#         refseedpoints,
#         parentspatialseedpoints,
#         refseedcellids,
#         levelset,
#         mergedmesh,
#         tol,
#         boundingradius,
#     )
#
# parentclosestpoints = CutCellDG.map_to_spatial(
#     refclosestpoints[1, :, :],
#     refclosestcellids[1, :],
#     CutCellDG.background_mesh(mergedmesh),
# )
# productclosestpoints = CutCellDG.map_to_spatial(
#     refclosestpoints[2, :, :],
#     refclosestcellids[2, :],
#     CutCellDG.background_mesh(mergedmesh),
# )
#
# @test allapprox(parentclosestpoints, productclosestpoints, 1e2eps())
#
# testcp = closest_point_on_plane(querypoints, normal, x0)
# @test allapprox(parentclosestpoints, testcp, 1e2eps())
#
#
# ################################################################################
#
#
#
# polyorder = 1
# delta = 0.05
# numqp = 2
# L, W = 2.0, 1.0
# basis = TensorProductBasis(2, polyorder)
# cgmesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], [2, 2], basis)
# dgmesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [2, 2], basis)
#
# normal = [1.0, 1.0] / sqrt(2)
# x0 = [1.0, 0.5 + delta]
# distancefunction(x) = plane_distance_function(x, normal, x0)
# levelset = CutCellDG.LevelSet(distancefunction, cgmesh, basis)
#
# cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
#
# cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
# facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)
# interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)
#
# mergedmesh = CutCellDG.MergedMesh(cutmesh, cellquads, facequads, interfacequads)
#
# refseedpoints, refseedcellids =
#     CutCellDG.seed_zero_levelset_with_interfacequads(interfacequads, mergedmesh)
# parentspatialseedpoints = CutCellDG.map_to_spatial(
#     refseedpoints[1, :, :],
#     refseedcellids[1, :],
#     CutCellDG.background_mesh(mergedmesh),
# )
# productspatialseedpoints = CutCellDG.map_to_spatial(
#     refseedpoints[2, :, :],
#     refseedcellids[2, :],
#     CutCellDG.background_mesh(mergedmesh),
# )
#
# @test allapprox(parentspatialseedpoints, productspatialseedpoints, 1e2eps())
#
# querypoints = CutCellDG.nodal_coordinates(cgmesh)
#
# tol = 1e-14
# boundingradius = 4.0
# refclosestpoints, refclosestcellids =
#     CutCellDG.closest_reference_points_on_merged_mesh(
#         querypoints,
#         refseedpoints,
#         parentspatialseedpoints,
#         refseedcellids,
#         levelset,
#         mergedmesh,
#         tol,
#         boundingradius,
#     )
#
# parentclosestpoints = CutCellDG.map_to_spatial(
#     refclosestpoints[1, :, :],
#     refclosestcellids[1, :],
#     CutCellDG.background_mesh(mergedmesh),
# )
# productclosestpoints = CutCellDG.map_to_spatial(
#     refclosestpoints[2, :, :],
#     refclosestcellids[2, :],
#     CutCellDG.background_mesh(mergedmesh),
# )
#
# @test allapprox(parentclosestpoints, productclosestpoints, 1e2eps())
#
# testcp = closest_point_on_plane(querypoints, normal, x0)
# @test allapprox(parentclosestpoints, testcp, 1e2eps())
#
#
# ################################################################################
#
# polyorder = 2
# delta = 0.1
# numqp = 2
# L, W = 4.0, 4.0
# basis = TensorProductBasis(2, polyorder)
# cgmesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], [2, 2], basis)
# dgmesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [2, 2], basis)
#
# center = [0.0, 0.0]
# radius = 2 * sqrt(2) + delta
# distancefunction(x) = circle_distance_function(x, center, radius)
# levelset = CutCellDG.LevelSet(distancefunction, cgmesh, basis)
#
# cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
#
# cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
# facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)
# interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)
#
# mergedmesh = CutCellDG.MergedMesh(cutmesh, cellquads, facequads, interfacequads)
#
# refseedpoints, refseedcellids =
#     CutCellDG.seed_zero_levelset_with_interfacequads(interfacequads, mergedmesh)
# parentspatialseedpoints = CutCellDG.map_to_spatial(
#     refseedpoints[1, :, :],
#     refseedcellids[1, :],
#     CutCellDG.background_mesh(mergedmesh),
# )
# productspatialseedpoints = CutCellDG.map_to_spatial(
#     refseedpoints[2, :, :],
#     refseedcellids[2, :],
#     CutCellDG.background_mesh(mergedmesh),
# )
#
# @test allapprox(parentspatialseedpoints, productspatialseedpoints, 1e2eps())
#
# querypoints = CutCellDG.nodal_coordinates(cgmesh)
#
# tol = 1e-14
# boundingradius = 4.5
# refclosestpoints, refclosestcellids =
#     CutCellDG.closest_reference_points_on_merged_mesh(
#         querypoints,
#         refseedpoints,
#         parentspatialseedpoints,
#         refseedcellids,
#         levelset,
#         mergedmesh,
#         tol,
#         boundingradius,
#     )
#
# parentclosestpoints = CutCellDG.map_to_spatial(
#     refclosestpoints[1, :, :],
#     refclosestcellids[1, :],
#     CutCellDG.background_mesh(mergedmesh),
# )
# productclosestpoints = CutCellDG.map_to_spatial(
#     refclosestpoints[2, :, :],
#     refclosestcellids[2, :],
#     CutCellDG.background_mesh(mergedmesh),
# )
#
# @test allapprox(parentclosestpoints, productclosestpoints, 0.2)
#
#
# testclosestpoints = closest_point_on_arc(querypoints, center, radius)
# checkpoints = deleteat!(collect(1:25), [1, 7, 13, 19, 25])
#
# v = testclosestpoints[:, checkpoints] - parentclosestpoints[:, checkpoints]
#
# @test allapprox(
#     testclosestpoints[:, checkpoints],
#     parentclosestpoints[:, checkpoints],
#     0.2,
# )
# @test allapprox(
#     testclosestpoints[:, checkpoints],
#     productclosestpoints[:, checkpoints],
#     0.2
# )
#
#
#
# ################################################################################
