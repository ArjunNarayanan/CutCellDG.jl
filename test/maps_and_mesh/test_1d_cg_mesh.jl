using CutCellDG
using Test
include("../useful_routines.jl")

x0 = [-1.0]
meshwidth = [2.0]
nelements = [3]
nodesperelement = 4

cgmesh = CutCellDG.CGMesh(x0,meshwidth,nelements,nodesperelement)
nc = CutCellDG.nodal_connectivity(cgmesh)
testnc = reshape([1,2,3,4,4,5,6,7,7,8,9,10],4,:)

@test allequal(nc,testnc)
