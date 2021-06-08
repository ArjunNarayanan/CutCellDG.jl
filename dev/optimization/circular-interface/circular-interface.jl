using Statistics
using PolynomialBasis
using ImplicitDomainQuadrature
using BenchmarkTools
# using Revise
using CutCellDG
include("../../../test/useful_routines.jl")
include("transformation-elasticity-solver.jl")

TES = TransformationElasticitySolver

polyorder = 2
nelmts = 15
penaltyfactor = 1e3


################################################################################
width = 1.0              # mm
interfacecenter = [0.5, 0.5]
interfaceradius = 0.4
outerradius = 1.2
################################################################################

################################################################################
K1, K2 = 247.0, 192.0    # GPa
mu1, mu2 = 126.0, 87.0   # GPa
lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)
stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)
rho1 = 3.93e3           # Kg/m^3
rho2 = 3.68e3           # Kg/m^3
V01 = 1.0 / rho1
V02 = 1.0 / rho2
theta0 = -0.067
ΔG0Jmol = -14351.0  # J/mol
molarmass = 0.147
ΔG0 = ΔG0Jmol / molarmass
transfstress =
    CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)
################################################################################



meshwidth = [width, width]
numqp = required_quadrature_order(polyorder) + 2

levelsetbasis = HermiteTensorProductBasis(2)
quad = tensor_product_quadrature(2, 4)
elasticitybasis = LagrangeTensorProductBasis(2, polyorder)

basispts = interpolation_points(elasticitybasis)
dim, numpts = size(interpolation_points(levelsetbasis))
cgmesh = CutCellDG.CGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], numpts)
dgmesh = CutCellDG.DGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basispts)

@btime levelset = CutCellDG.LevelSet(
    x -> -circle_distance_function(x, interfacecenter, interfaceradius)[1],
    cgmesh,
    levelsetbasis,
    quad,
)

elementsize = CutCellDG.element_size(cgmesh)
minelmtsize = minimum(elementsize)
maxelmtsize = maximum(elementsize)
penalty = penaltyfactor / minelmtsize * 0.5 * (lambda1 + mu1 + lambda2 + mu2)
tol = minelmtsize^(polyorder + 1)
boundingradius = 1.5 * maxelmtsize

cutmesh = CutCellDG.CutMesh(dgmesh, levelset)

cellquads = CutCellDG.CellQuadratures(cutmesh,levelset,numqp)
@btime interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)
@btime facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)
