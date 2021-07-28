using PolynomialBasis
using Test
# using Revise
using CutCellDG
include("../useful_routines.jl")

polyorder = 1
numqp = 2

basis = LagrangeTensorProductBasis(2,1)
dim,numpts = size(interpolation_points(basis))
mesh = CutCellDG.CGMesh([0.0, 0.0], [3.0, 1.0], [3, 1], numpts)

x0 = [1.5,0.0]
normal = [1.,0.]
distancefunc(x) = plane_distance_function(x,normal,x0)
levelset = CutCellDG.LevelSet(distancefunc,mesh,basis)
CutCellDG.load_coefficients!(levelset,1)
@test allapprox(levelset.interpolater.coeffs,[-1.5,-1.5,-0.5,-0.5]')
@test levelset([0.,0.]) ≈ -1.0

CutCellDG.update_coefficients!(levelset,2,[-0.6,-0.6,0.4,0.4])
@test allapprox(CutCellDG.coefficients(levelset,2),[-0.6,-0.6,0.4,0.4])
@test allapprox(CutCellDG.coefficients(levelset,1),[-1.5,-1.5,-0.6,-0.6])
@test allapprox(CutCellDG.coefficients(levelset,3),[0.4,0.4,1.5,1.5])


basis = HermiteTensorProductBasis(2)
numqp = required_quadrature_order(3)
quad = tensor_product_quadrature(2,numqp)
dim,numpts = size(interpolation_points(basis))
mesh = CutCellDG.CGMesh([0.,0.],[1.,1.],[2,2],numpts)
x0 = [0.75,0.0]
normal = [1.0,0.0]
distancefunc(x) = plane_distance_function(x,normal,x0)

levelset = CutCellDG.LevelSet(distancefunc,mesh,basis)

coeffs1 = CutCellDG.coefficients(levelset,1)
testcoeffs = [-0.75  -0.75  -0.25  -0.25
               0.0    0.0    0.0    0.0
               0.25   0.25   0.25   0.25
               0.0    0.0    0.0    0.0]
testcoeffs = vec(testcoeffs)
@test allapprox(coeffs1,testcoeffs,1e3eps())

x0 = [1.0,0.5]
normal = [1.0,1.0]/sqrt(2)

function testcubic(v)
    x,y = v
    return 8x^3*y^3 + 4x^2*y^3 + 17*x*y + 11
end

levelset = CutCellDG.LevelSet(testcubic,mesh,basis)
testcellid = 4
CutCellDG.load_coefficients!(levelset,testcellid)
cellmap = CutCellDG.cell_map(mesh,testcellid)
xrange = range(-1,stop=1,length=5)
testp = ImplicitDomainQuadrature.tensor_product_points(xrange',xrange')
spatialtestp = cellmap(testp)

vals = vec(mapslices(levelset,testp,dims=1))
testvals = vec(mapslices(testcubic,spatialtestp,dims=1))
@test allapprox(vals,testvals,1e4eps())
