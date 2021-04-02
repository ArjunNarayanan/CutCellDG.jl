using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../useful_routines.jl")

polyorder = 1
numqp = 2

basis = TensorProductBasis(2, polyorder)
x0 = [0.0, 0.0]
meshwidths = [3.0, 1.0]
nelements = [3, 1]
dgmesh = CutCellDG.DGMesh(x0, meshwidths, nelements, basis)
cgmesh = CutCellDG.CGMesh(x0, meshwidths, nelements, basis)

normal = [1.0, 0.0]
xI = [0.5, 0.0]
levelset = CutCellDG.LevelSet(x->plane_distance_function(x,normal,xI),cgmesh,basis)

cutmesh = CutCellDG.CutMesh(dgmesh, levelset)

interfacequads =
    CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)
@test size(interfacequads.quads) == (2, 1)
testcelltoquad = [1, 0, 0]
@test allequal(interfacequads.celltoquad, testcelltoquad)

testnormals = repeat(normal, inner = (1, numqp))
@test allapprox(testnormals, CutCellDG.interface_normals(interfacequads, 1))
@test interfacequads[1, 1] ≈ interfacequads[-1, 1]

testtangents = repeat([0.0, 1.0], inner = (1, numqp))
testscaleareas = repeat([0.5], numqp)
@test allapprox(testtangents, CutCellDG.interface_tangents(interfacequads, 1))
@test allapprox(
    testscaleareas,
    CutCellDG.interface_scale_areas(interfacequads, 1),
)
