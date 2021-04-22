using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../useful_routines.jl")

L, W = 1.0, 1.0
basis = TensorProductBasis(2, 1)

mesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], [1, 1], basis)
levelset = CutCellDG.LevelSet(
    x -> plane_distance_function(x, [1.0, 0.0], [0.75, 0.0]),
    mesh,
    basis,
)

refseedpoints, refseedcellids =
    CutCellDG.seed_zero_levelset(2, levelset, [1], 1e3eps(), 4.5)
testseedpoints = [0.5 0.5 0.5 0.5
                 -1/3 1/3 -1/3 1/3]
testseedcellids = repeat([1],4)

@test allapprox(testseedpoints,refseedpoints)
@test allequal(testseedcellids,refseedcellids)




L, W = 2.0, 2.0
basis = TensorProductBasis(2, 2)

mesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], [2, 2], basis)
levelset = CutCellDG.LevelSet(
    x -> plane_distance_function(x, [1.0, 0.0], [1.5, 0.0]),
    mesh,
    basis,
)
cutmesh = CutCellDG.CutMesh(mesh,levelset)

refseedpoints, refseedcellids =
    CutCellDG.seed_zero_levelset(2, levelset, cutmesh)

testseedpoints = [0.   0.  0.   0.  0.   0.  0.   0.
                  -1/3 1/3 -1/3 1/3 -1/3 1/3 -1/3 1/3]
testseedcellids = [3,3,3,3,4,4,4,4]
@test allapprox(testseedpoints,refseedpoints,1e3eps())
@test allequal(refseedcellids, testseedcellids)


refseedpoints,refseedcellids = CutCellDG.seed_zero_levelset(2,levelset,[1,2],1e-12,1.0)
@test isempty(refseedpoints)
@test isempty(refseedcellids)
