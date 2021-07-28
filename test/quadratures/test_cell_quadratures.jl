using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../useful_routines.jl")

polyorder = 1
numqp = 2
elasticitybasis = LagrangeTensorProductBasis(2, polyorder)
levelsetbasis = HermiteTensorProductBasis(2)
levelset = InterpolatingPolynomial(1, levelsetbasis)

x0 = [0.0, 0.0]
meshwidths = [3.0, 1.0]
nelements = [3, 1]
dim,numinterp = size(interpolation_points(levelsetbasis))
points = interpolation_points(elasticitybasis)
dgmesh = CutCellDG.DGMesh(x0, meshwidths, nelements, points)
cgmesh = CutCellDG.CGMesh(x0, meshwidths, nelements, numinterp)

normal = [1.0, 0.0]
xI = [0.5, 0.0]
tol = 1e-3
perturbation = 1e-3

levelset = CutCellDG.LevelSet(
    x -> plane_distance_function(x, normal, xI),
    cgmesh,
    levelsetbasis,
)

cutmesh = CutCellDG.CutMesh(dgmesh, levelset)

cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
@test length(cellquads.quads) == 3
testcelltoquad = [
    2 1 1
    3 0 0
]
@test allequal(cellquads.celltoquad, testcelltoquad)
