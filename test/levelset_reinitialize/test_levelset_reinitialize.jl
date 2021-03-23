using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("../useful_routines.jl")


polyorder = 2
nelmts = 10
L, W = 1.0, 1.0
basis = TensorProductBasis(2, polyorder)
numqp = required_quadrature_order(polyorder) + 2
quad = tensor_product_quadrature(2, numqp)
levelset = InterpolatingPolynomial(1, basis)

mesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
levelsetcoeffs = CutCell.levelset_coefficients(distancefunction, mesh)
cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)

refseedpoints, spatialseedpoints, seedcellids =
    CutCell.seed_zero_levelset(2, levelset, levelsetcoeffs, cutmesh)
