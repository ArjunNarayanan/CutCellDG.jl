using PolynomialBasis
using ImplicitDomainQuadrature
using Test
using Revise
using CutCellDG
include("../useful_routines.jl")

testfunc(x) = [x[1]^3 + 3x[2]^3 + 2x[1]^2 * x[2] + 8x[1] * x[2]]

cellmap = CutCellDG.CellMap([-1.0, -1.0], [1.0, 1.0])
detjac = CutCellDG.determinant_jacobian(cellmap)
basis = HermiteTensorProductBasis(2)
quad = tensor_product_quadrature(2, 4)

M = CutCellDG.mass_matrix(basis, quad, 1, 1.0)
R = CutCellDG.linear_form(testfunc, basis, quad, cellmap, 1, detjac)
C = M\R

interp = InterpolatingPolynomial(1,basis)
update!(interp,C)

xrange = range(-1,stop=1,length=10)
points = ImplicitDomainQuadrature.tensor_product_points(xrange',xrange')
vals = mapslices(interp,points,dims=1)
testvals = mapslices(testfunc,points,dims=1)
@test allapprox(vals,testvals)
