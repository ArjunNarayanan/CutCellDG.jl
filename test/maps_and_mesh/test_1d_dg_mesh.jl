using PolynomialBasis
using CartesianMesh
using CutCellDG
using Test
include("../useful_routines.jl")

x0 = [-1.0]
meshwidth = [2.0]
nelements = [3]
polyorder = 3
solverbasis = LagrangeTensorProductBasis(1, polyorder)
refpoints = interpolation_points(solverbasis)

dgmesh = CutCellDG.DGMesh(x0,meshwidth,nelements,refpoints)
nc = CutCellDG.nodal_connectivity(dgmesh)
testnc = reshape(1:12,4,:)
@test allequal(nc,testnc)
