using PolynomialBasis
# using Revise
using CutCellDG
include("../useful_routines.jl")

polyorder = 1
numqp = 2

basis = TensorProductBasis(2, polyorder)
levelset = InterpolatingPolynomial(1, basis)
mesh = CutCellDG.CGMesh([0.0, 0.0], [3.0, 1.0], [3, 1], basis)
nodalcoordinates = CutCellDG.nodal_coordinates(mesh)

x0 = [1.5,0.0]
normal = [1.,0.]
levelset = CutCellDG.LevelSet(x->plane_distance_function(x,normal,x0),mesh,basis)
CutCellDG.load_coefficients!(levelset,1)
@test allapprox(levelset.interpolater.coeffs,[-1.5,-1.5,-0.5,-0.5]')
@test levelset([0.,0.]) ≈ -1.0

CutCellDG.update_coefficients!(levelset,2,[-0.6,-0.6,0.4,0.4])
@test allapprox(CutCellDG.coefficients(levelset,2),[-0.6,-0.6,0.4,0.4])
@test allapprox(CutCellDG.coefficients(levelset,1),[-1.5,-1.5,-0.6,-0.6])
@test allapprox(CutCellDG.coefficients(levelset,3),[0.4,0.4,1.5,1.5])
