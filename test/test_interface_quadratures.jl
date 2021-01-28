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
levelsetcoeffs = CutCellDG.levelset_coefficients(x -> plane_distance_function(x, normal, x0),mesh)

cutmesh = CutCellDG.CutMesh(mesh,levelset,levelsetcoeffs)

interfacequads = CutCellDG.InterfaceQuadratures(
    cutmesh,
    levelset,
    levelsetcoeffs,
    numqp,
)
@test size(interfacequads.quads) == (2,1)
testcelltoquad = [1, 0, 0]
@test allequal(interfacequads.celltoquad, testcelltoquad)

testnormals = repeat(normal,inner=(1,numqp))
@test allapprox(testnormals,CutCellDG.interface_normals(interfacequads,1))

@test interfacequads[1,1] ≈ interfacequads[-1,1]
