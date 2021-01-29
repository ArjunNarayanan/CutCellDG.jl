using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("useful_routines.jl")

polyorder = 1
numqp = 2

basis = TensorProductBasis(2, polyorder)
levelset = InterpolatingPolynomial(1, basis)
mesh = CutCellDG.DGMesh([0.0, 0.0], [2.0, 1.0], [2, 1], basis)

normal = [1.0, 0.0]
x0 = [1.1, 0.0]
levelsetcoeffs =
    CutCellDG.levelset_coefficients(x -> plane_distance_function(x, normal, x0), mesh)

cutmesh = CutCellDG.CutMesh(mesh, levelset, levelsetcoeffs)
cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
facequads = CutCellDG.FaceQuadratures(cutmesh,levelset,levelsetcoeffs,numqp)

mergedwithcell = CutCellDG.merge_tiny_cells_in_mesh!(cellquads,facequads,interfacequads,cutmesh)
mergedmesh = CutCellDG.MergedMesh(cutmesh,mergedwithcell)

lambda, mu = (1.0, 2.0)
penalty = 1.0
dx = 0.1
e11 = dx / 2.0
e22 = -lambda / (lambda + 2mu) * e11
dy = e22
s11 = (lambda + 2mu) * e11 + lambda * e22

stiffness = CutCell.HookeStiffness(lambda, mu, lambda, mu)