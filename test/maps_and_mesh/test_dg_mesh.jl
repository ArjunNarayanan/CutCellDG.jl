using Test
using CartesianMesh
using PolynomialBasis
# using Revise
using CutCellDG
include("../useful_routines.jl")


basis = LagrangeTensorProductBasis(2,1)
mesh = UniformMesh([0.,0.],[2.,1.],[2,2])
cellmaps = CutCellDG.construct_cell_maps(mesh)

points = interpolation_points(basis)
nodalcoordinates = CutCellDG.dg_nodal_coordinates(cellmaps,points)
testnodalcoordinates = [0.  0.  1.  1.  0.  0.  1.  1.  1.  1.  2.  2.  1.  1.  2.  2.
                        0.  .5  0.  .5  .5  1.  .5  1.  0.  .5  0.  .5  .5  1.  .5  1.]
@test allapprox(nodalcoordinates,testnodalcoordinates)

nodalconnectivity = CutCellDG.dg_nodal_connectivity(4,4)
testnodalconnectivity = [1  5  9   13
                         2  6  10  14
                         3  7  11  15
                         4  8  12  16]
@test allequal(nodalconnectivity,testnodalconnectivity)



basis = LagrangeTensorProductBasis(2,2)
points = interpolation_points(basis)
dgmesh = CutCellDG.DGMesh([0.,0.],[2.,1.],[2,2],points)
nodalcoordinates = CutCellDG.nodal_coordinates(dgmesh)
tn1 = [0.  0.   0.  0.5  0.5  0.5  1.  1.   1.  0.  0.   0.  0.5  0.5  0.5  1.  1.  1.
       0.  0.25 0.5 0.   0.25 0.5  0.  0.25 0.5 0.5 0.75 1.  0.5  0.75 1.   0.5 0.75 1.]
tn2 = copy(tn1)
tn2[1,:] .+= 1.0
testnodalcoordinates = hcat(tn1,tn2)
@test allapprox(nodalcoordinates,testnodalcoordinates)

nodalconnectivity = hcat([CutCellDG.nodal_connectivity(dgmesh,i) for i = 1:4]...)
testnodalconnectivity = reshape(1:36,9,:)
@test allequal(nodalconnectivity,testnodalconnectivity)
