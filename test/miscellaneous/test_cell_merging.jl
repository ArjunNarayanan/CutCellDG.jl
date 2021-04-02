using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../useful_routines.jl")


p = [0.0, 0.0]

mergemapper = CutCellDG.MergeMapper()

south = CutCellDG.cell_map_to_south()
@test allapprox(south(p), [0.0, -2.0])
@test allapprox(mergemapper[1](p), [0.0, -2.0])

southeast = CutCellDG.cell_map_to_south_east()
@test allapprox(southeast(p), [2.0, -2.0])
@test allapprox(mergemapper[5](p), [2.0, -2.0])

east = CutCellDG.cell_map_to_east()
@test allapprox(east(p), [2.0, 0.0])
@test allapprox(mergemapper[2](p), [2.0, 0.0])

northeast = CutCellDG.cell_map_to_north_east()
@test allapprox(northeast(p), [2.0, 2.0])
@test allapprox(mergemapper[6](p), [2.0, 2.0])

north = CutCellDG.cell_map_to_north()
@test allapprox(north(p), [0.0, 2.0])
@test allapprox(mergemapper[3](p), [0.0, 2.0])

northwest = CutCellDG.cell_map_to_north_west()
@test allapprox(northwest(p), [-2.0, 2.0])
@test allapprox(mergemapper[7](p), [-2.0, 2.0])

west = CutCellDG.cell_map_to_west()
@test allapprox(west(p), [-2.0, 0.0])
@test allapprox(mergemapper[4](p), [-2.0, 0.0])

southwest = CutCellDG.cell_map_to_south_west()
@test allapprox(southwest(p), [-2.0, -2.0])
@test allapprox(mergemapper[8](p), [-2.0, -2.0])

polyorder = 1
numqp = 2

basis = TensorProductBasis(2, polyorder)
x0 = [0.0, 0.0]
meshwidths = [2.0, 1.0]
nelements = [2, 1]
dgmesh = CutCellDG.DGMesh(x0, meshwidths, nelements, basis)
cgmesh = CutCellDG.CGMesh(x0, meshwidths, nelements, basis)


normal = [1.0, 0.0]
xI = [1.1, 0.0]
levelset = CutCellDG.LevelSet(x->plane_distance_function(x,normal,xI),cgmesh,basis)

cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
interfacequads =
    CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)
facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)

mergedwithcell, hasmergedcells = CutCellDG.merge_tiny_cells_in_mesh!(
    cutmesh,
    cellquads,
    facequads,
    interfacequads,
)
mergedmesh = CutCellDG.MergedMesh(cutmesh, mergedwithcell)
@test CutCellDG.number_of_nodes(mergedmesh) == 8
@test hasmergedcells
