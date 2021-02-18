using Test
using CartesianMesh
# using Revise
using CutCellDG
include("../useful_routines.jl")

cm = CutCellDG.CellMap([0.,2.],[3.,3.])
@test CutCellDG.dimension(cm) == 2
@test CutCellDG.jacobian(cm) == [1.5,0.5]
@test allapprox(cm([0.,0.]),[1.5,2.5])
@test allapprox(cm([1.,0.]),[3.,2.5])

mesh = UniformMesh([0.,0.],[1.,1.],[2,2])
@test allequal(CutCellDG.elements_per_mesh_side(mesh),[2,2])
cellmaps = CutCellDG.construct_cell_maps(mesh)
cm1 = CutCellDG.CellMap([0.,0.],[0.5,0.5])
cm2 = CutCellDG.CellMap([0.,0.5],[0.5,1.])
cm3 = CutCellDG.CellMap([0.5,0.],[1.,0.5])
cm4 = CutCellDG.CellMap([0.5,0.5],[1.,1.])
@test allequal(cellmaps,[cm1,cm2,cm3,cm4])

cm = CutCellDG.CellMap([0.,0.],[1.,1.])
@test CutCellDG.determinant_jacobian(cm) ≈ 0.25

cm = CutCellDG.CellMap([1.,1.],[5.,8.])
@test CutCellDG.determinant_jacobian(cm) ≈ 7.0

cm = CutCellDG.CellMap([2.,1.],[4.,2.])
@test allapprox(CutCellDG.inverse(cm,[3,1.5]),[0.,0.])
@test allapprox(CutCellDG.inverse(cm,[4,2.]),[1.,1.])
@test allapprox(CutCellDG.inverse(cm,[2,1]),[-1.,-1.])
