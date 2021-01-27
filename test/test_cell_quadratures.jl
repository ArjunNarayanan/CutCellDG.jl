using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("useful_routines.jl")

polyorder = 1
numqp = 2
basis = TensorProductBasis(2, polyorder)
levelset = InterpolatingPolynomial(1, basis)
mesh = CutCellDG.DGMesh([0.0, 0.0], [3.0, 1.0], [3, 1], basis)

normal = [1.0, 0.0]
x0 = [0.5, 0.0]
tol = 1e-3
perturbation = 1e-3
levelsetcoeffs = CutCellDG.levelset_coefficients(x -> plane_distance_function(x, normal, x0),mesh)

cutmesh = CutCellDG.CutMesh(mesh,levelset,levelsetcoeffs)

cellquads = CutCellDG.CellQuadratures(
    cutmesh,
    levelset,
    levelsetcoeffs,
    numqp,
    numqp
)
@test length(cellquads.quads) == 3
testcelltoquad = [
    2 1 1
    3 0 0
]
@test allequal(cellquads.celltoquad, testcelltoquad)
