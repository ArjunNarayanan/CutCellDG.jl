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

x0 = [0.0, 0.0]
meshwidths = [3.0, 1.0]
nelements = [3, 1]
points = interpolation_points(elasticitybasis)
dim, nf = size(interpolation_points(levelsetbasis))
dgmesh = CutCellDG.DGMesh(x0, meshwidths, nelements, points)
cgmesh = CutCellDG.CGMesh(x0, meshwidths, nelements, nf)

normal = [1.0, 0.0]
xI = [0.5, 0.0]
levelset = CutCellDG.LevelSet(
    x -> plane_distance_function(x, normal, xI),
    cgmesh,
    levelsetbasis,
)
cutmesh = CutCellDG.CutMesh(dgmesh, levelset)

interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)
@test size(interfacequads.quads) == (2, 1)
testcelltoquad = [1, 0, 0]
@test allequal(interfacequads.celltoquad, testcelltoquad)

testnormals = repeat(normal, inner = (1, numqp))
@test allapprox(
    testnormals,
    CutCellDG.interface_normals(interfacequads, 1),
    1e3eps(),
)
@test interfacequads[1, 1] ≈ interfacequads[-1, 1]

testtangents = repeat([0.0, 1.0], inner = (1, numqp))
testscaleareas = repeat([0.5], numqp)
@test allapprox(
    testtangents,
    CutCellDG.interface_tangents(interfacequads, 1),
    1e3eps(),
)
@test allapprox(
    testscaleareas,
    CutCellDG.interface_scale_areas(interfacequads, 1),
)
