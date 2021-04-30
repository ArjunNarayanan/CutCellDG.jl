using Test
using PyPlot
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../../../test/useful_routines.jl")

meshwidth = [1.,1.]
polyorder = 1
nelmts = 5
penaltyfactor = 1e2
numqp = required_quadrature_order(polyorder) + 2

K1, K2 = 247.0, 192.0    # GPa
mu1, mu2 = 126.0, 87.0   # GPa

lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)
stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)

rho1 = 3.93e3           # Kg/m^3
rho2 = 3.68e3           # Kg/m^3
V01 = 1.0 / rho1
V02 = 1.0 / rho2

ΔG0Jmol = -14351.0  # J/mol
molarmass = 0.147
ΔG0 = ΔG0Jmol / molarmass

theta0 = -0.067
transfstress =
    CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)

basis = TensorProductBasis(2, polyorder)
interfacepos0 = 0.9
outerradius = 1.2
CFL = 0.25

cgmesh = CutCellDG.CGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basis)
dgmesh = CutCellDG.DGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basis)
levelset = CutCellDG.LevelSet(
    x -> plane_distance_function(x, [1.0,0.0], [interfacepos0,0.0]),
    cgmesh,
    basis,
)
coeffs0 = copy(CutCellDG.coefficients(levelset))
paddedmesh = CutCellDG.BoundaryPaddedMesh(cgmesh, 1)

nodalcoordinates = CutCellDG.nodal_coordinates(cgmesh)
elementsize = CutCellDG.element_size(cgmesh)
minelmtsize = minimum(elementsize)
maxelmtsize = maximum(elementsize)
penalty = penaltyfactor / minelmtsize * 0.5 * (lambda1 + mu1 + lambda2 + mu2)
tol = minelmtsize^(polyorder + 1)
boundingradius = 1.5 * maxelmtsize

ycoords = nodalcoordinates[2,:]
sortidx = sortperm(ycoords)
ycoords = ycoords[sortidx]
