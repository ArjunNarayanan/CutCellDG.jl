using Test
using SparseArrays
using Revise
using CutCellDG
include("useful_routines.jl")


@test CutCellDG.node_to_dof_id(9,1,2) == 17
@test CutCellDG.node_to_dof_id(5,2,2) == 10

nodeids = [14,15,16,19,20,21,24,25,26]
edofs = CutCellDG.element_dofs(nodeids,2)
testedofs = [27,28,29,30,31,32,37,38,39,40,41,42,47,48,49,50,51,52]
@test allequal(edofs,testedofs)

rows,cols = CutCellDG.element_dofs_to_operator_dofs(1:4,[5,6,7])
testrows = vcat(1:4,1:4,1:4)
@test allequal(testrows,rows)
testcols = vcat([5,5,5,5],[6,6,6,6],[7,7,7,7])
@test allequal(testcols,cols)
