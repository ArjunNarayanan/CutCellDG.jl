using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../useful_routines.jl")

x0 = [0.0, 0.0]
widths = [1.0, 1.0]
nelements = [2, 2]
polyorder = 2
numqp = required_quadrature_order(polyorder) + 2

elasticitybasis = LagrangeTensorProductBasis(2, polyorder)
levelsetbasis = HermiteTensorProductBasis(2)
quad = tensor_product_quadrature(2,4)
refpoints = interpolation_points(elasticitybasis)
dim,nf = size(interpolation_points(levelsetbasis))

dgmesh = CutCellDG.DGMesh(x0,widths,nelements,refpoints)
CutCellDG.make_vertical_periodic!(dgmesh)

testcellconnectivity = [2  1  4  3
                        3  4  0  0
                        2  1  4  3
                        0  0  1  2]
@test allequal(testcellconnectivity,CutCellDG.cell_connectivity(dgmesh,:,:))
