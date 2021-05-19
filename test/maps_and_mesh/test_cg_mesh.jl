using Test
using CartesianMesh
using PolynomialBasis
import ImplicitDomainQuadrature: tensor_product_points
# using Revise
using CutCellDG
include("../useful_routines.jl")

mesh = UniformMesh([0.,0.],[1.,1.],[2,2])
@test allequal(CutCellDG.elements_per_mesh_side(mesh),[2,2])

coords = CutCellDG.cg_nodal_coordinates(mesh.x0,mesh.widths,mesh.nelements,[5,5])
xrange = range(0.0,stop=1.,length=5)
ycoords = repeat(xrange,5)
xcoords = repeat(xrange,inner=5)
@test allapprox(coords,vcat(xcoords',ycoords'))

connectivity = CutCellDG.cg_nodal_connectivity([5,5],3,9,mesh.nelements)
c1 = [1,2,3,6,7,8,11,12,13]
c2 = [3,4,5,8,9,10,13,14,15]
c3 = [11,12,13,16,17,18,21,22,23]
c4 = [13,14,15,18,19,20,23,24,25]
testconn = hcat(c1,c2,c3,c4)
@test allequal(connectivity,testconn)

mesh = UniformMesh([0.,0.],[1.,1.],[3,2])
connectivity = CutCellDG.cg_nodal_connectivity([10,7],4,16,mesh.nelements)
testnodeconn = vcat(43:46,50:53,57:60,64:67)
@test allequal(connectivity[:,5],testnodeconn)

mesh = UniformMesh([0.,0.],[1.,1.],[3,2])
femesh = CutCellDG.CGMesh(mesh,16)
testnodeconn = vcat(43:46,50:53,57:60,64:67)
@test allequal(testnodeconn,CutCellDG.nodal_connectivity(femesh,5))
@test CutCellDG.nodes_per_element(femesh) == 16
@test CutCellDG.nodes_per_mesh_side(femesh) == [10,7]
@test CutCellDG.number_of_nodes(femesh) == 70

mesh = CutCellDG.CGMesh([2.,1.],[2.,1.],[3,2],9)
@test CutCellDG.number_of_cells(mesh) == 6
@test CutCellDG.number_of_nodes(mesh) == 35
x1 = range(2.0,stop=4.0,length=7)
x2 = range(1.0,stop=2.0,length=5)
testpoints = tensor_product_points(x1',x2')
@test allapprox(CutCellDG.nodal_coordinates(mesh),testpoints)
testconnectivity = [1 2 3 6 7 8 11 12 13
                    3 4 5 8 9 10 13 14 15
                    11 12 13 16 17 18 21 22 23
                    13 14 15 18 19 20 23 24 25
                    21 22 23 26 27 28 31 32 33
                    23 24 25 28 29 30 33 34 35]'
@test all([allequal(CutCellDG.nodal_connectivity(mesh,i),testconnectivity[:,i]) for i = 1:6])
