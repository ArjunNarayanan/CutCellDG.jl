using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../useful_routines.jl")

polyorder = 1
numqp = 2

basis = TensorProductBasis(2, polyorder)
levelset = InterpolatingPolynomial(1, basis)
mesh = CutCellDG.DGMesh([0.0, 0.0], [3.0, 1.0], [3, 1], basis)
nodalcoordinates = CutCellDG.nodal_coordinates(mesh)

normal = [1.0, 0.0]
x0 = [0.5, 0.0]
tol = 1e-3
perturbation = 0.0
levelsetcoeffs = plane_distance_function(nodalcoordinates, normal, x0)

cellsign =
    CutCellDG.cell_sign!(mesh, levelset, levelsetcoeffs, tol, perturbation)
@test allequal(cellsign, [0, 1, 1])

normal = [1.0, 0.0]
x0 = [1.0, 0.0]
tol = 1e-3
perturbation = 1e-3
levelsetcoeffs = plane_distance_function(nodalcoordinates, normal, x0)
cellsign =
    CutCellDG.cell_sign!(mesh, levelset, levelsetcoeffs, tol, perturbation)

@test allapprox(
    levelsetcoeffs[1:4],
    [-1.0 + perturbation, -1.0 + perturbation, perturbation, perturbation],
)
@test allapprox(
    levelsetcoeffs[5:8],
    [perturbation, perturbation, 1.0 + perturbation, 1.0 + perturbation],
)
@test allapprox(levelsetcoeffs[9:12], [1.0, 1.0, 2.0, 2.0])

posactivenodeids = CutCellDG.active_node_ids(mesh,+1,cellsign)
@test allequal(posactivenodeids, 1:12)
negactivenodeids = CutCellDG.active_node_ids(mesh,-1, cellsign)
@test allequal(negactivenodeids, 1:4)

totalnumnodes = CutCellDG.number_of_nodes(mesh)
cutmeshnodeids =
    CutCellDG.cut_mesh_nodeids(posactivenodeids, negactivenodeids, totalnumnodes)
testcutmeshnodeids = [
    1 2 3 4 5 6 7 8 9 10 11 12
    13 14 15 16 0 0 0 0 0 0 0 0
]
@test allequal(cutmeshnodeids, testcutmeshnodeids)


cutmesh = CutCellDG.CutMesh(mesh, levelset, levelsetcoeffs)
@test allequal(CutCellDG.nodal_connectivity(cutmesh, +1, 1), [1, 2, 3, 4])
@test allequal(CutCellDG.nodal_connectivity(cutmesh, -1, 1), [13, 14, 15, 16])
