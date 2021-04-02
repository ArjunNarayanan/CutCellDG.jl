using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("../useful_routines.jl")

polyorder = 1
numqp = 2
basis = TensorProductBasis(2, polyorder)
levelset = InterpolatingPolynomial(1, basis)
x0 = [0.,0.]
meshwidths = [3.,1.]
nelements = [3,1]
dgmesh = CutCellDG.DGMesh(x0, meshwidths, nelements, basis)
cgmesh = CutCellDG.CGMesh(x0,meshwidths,nelements,basis)

normal = [1.0, 0.0]
xI = [0.5, 0.0]
tol = 1e-3
perturbation = 1e-3
levelset = CutCellDG.LevelSet(x->plane_distance_function(x,normal,xI),cgmesh,basis)

cutmesh = CutCellDG.CutMesh(dgmesh,levelset)

cellquads = CutCellDG.CellQuadratures(
    cutmesh,
    levelset,
    numqp,
)
@test length(cellquads.quads) == 3
testcelltoquad = [
    2 1 1
    3 0 0
]
@test allequal(cellquads.celltoquad, testcelltoquad)
