using Test
using PolynomialBasis
# using Revise
using CutCellDG
include("../useful_routines.jl")

x0 = [1.0, 5.0]
L, W = 4.0, 1.0
nelmtsx = 2
nelmtsy = 1
numghostlayers = 2
nodesperelement = 9

mesh = CutCellDG.CGMesh(x0, [L, W], [nelmtsx, nelmtsy], nodesperelement)
paddedmesh = CutCellDG.BoundaryPaddedMesh(mesh, numghostlayers)

dx = 1.0
dy = 0.5

testbottomcoords = [
    1.0 1.0 2.0 2.0 3.0 3.0 4.0 4.0 5.0 5.0
    4.0 4.5 4.0 4.5 4.0 4.5 4.0 4.5 4.0 4.5
]
testrightcoords = [
    6.0 6.0 6.0 7.0 7.0 7.0
    5.0 5.5 6.0 5.0 5.5 6.0
]
testtopcoords = [
    1.0 1.0 2.0 2.0 3.0 3.0 4.0 4.0 5.0 5.0
    6.5 7.0 6.5 7.0 6.5 7.0 6.5 7.0 6.5 7.0
]
testleftcoords = [
    -1.0 -1.0 -1.0 0.0 0.0 0.0
    5.0 5.5 6.0 5.0 5.5 6.0
]

@test allapprox(
    CutCellDG.bottom_ghost_coordinates(paddedmesh),
    testbottomcoords,
)
@test allapprox(CutCellDG.right_ghost_coordinates(paddedmesh), testrightcoords)
@test allapprox(CutCellDG.top_ghost_coordinates(paddedmesh), testtopcoords)
@test allapprox(CutCellDG.left_ghost_coordinates(paddedmesh), testleftcoords)

x0 = [0.0, 0.0]
L, W = 2.0, 1.0
nelmtsx, nelmtsy = 2, 1
numghostlayers = 1
polyorder = 2

xI = [0.0, 0.5]
normal = [0.0, 1.0]
tol = 1e-8
boundingradius = 6.0

basis = TensorProductBasis(2, polyorder)
mesh = CutCellDG.CGMesh(x0, [L, W], [nelmtsx, nelmtsy], basis)

levelset =
    CutCellDG.LevelSet(x -> plane_distance_function(x, normal, xI), mesh, basis)

paddedmesh = CutCellDG.BoundaryPaddedMesh(
    CutCellDG.background_mesh(levelset),
    numghostlayers,
)
cutmesh = CutCellDG.CutMesh(mesh, levelset)

refseedpoints, seedcellids = CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
spatialseedpoints =
    CutCellDG.map_to_spatial(refseedpoints, seedcellids, cutmesh)

paddedlevelset = CutCellDG.BoundaryPaddedLevelSet(
    paddedmesh,
    refseedpoints,
    seedcellids,
    spatialseedpoints,
    levelset,
    tol,
    boundingradius,
)

Dmx = CutCellDG.first_order_horizontal_backward_difference(paddedlevelset)
Dpx = CutCellDG.first_order_horizontal_forward_difference(paddedlevelset)
Dmy = CutCellDG.first_order_vertical_backward_difference(paddedlevelset)
Dpy = CutCellDG.first_order_vertical_forward_difference(paddedlevelset)

numnodes = CutCellDG.number_of_nodes(mesh)
@test allapprox(Dmy, ones(length(numnodes)))
@test allapprox(Dpy, ones(length(numnodes)))
@test allapprox(Dmx, zeros(length(numnodes)))
@test allapprox(Dpx, zeros(length(numnodes)))



x0 = [0.0, 0.0]
L, W = 2.0, 1.0
nelmtsx, nelmtsy = 2, 1
numghostlayers = 1
polyorder = 2

xI = [0.5, 0.0]
normal = [1.0, 0.0]
tol = 1e-8
boundingradius = 6.0

basis = TensorProductBasis(2, polyorder)
mesh = CutCellDG.CGMesh(x0, [L, W], [nelmtsx, nelmtsy], basis)
numnodes = CutCellDG.number_of_nodes(mesh)

levelset =
    CutCellDG.LevelSet(x -> plane_distance_function(x, normal, xI), mesh, basis)
paddedmesh = CutCellDG.BoundaryPaddedMesh(
    CutCellDG.background_mesh(levelset),
    numghostlayers,
)
cutmesh = CutCellDG.CutMesh(mesh, levelset)


refseedpoints, seedcellids =
    CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
spatialseedpoints =
    CutCellDG.map_to_spatial(refseedpoints, seedcellids, cutmesh)


paddedlevelset = CutCellDG.BoundaryPaddedLevelSet(
    paddedmesh,
    refseedpoints,
    seedcellids,
    spatialseedpoints,
    levelset,
    tol,
    boundingradius,
)

Dmx = CutCellDG.first_order_horizontal_backward_difference(paddedlevelset)
Dpx = CutCellDG.first_order_horizontal_forward_difference(paddedlevelset)
Dmy = CutCellDG.first_order_vertical_backward_difference(paddedlevelset)
Dpy = CutCellDG.first_order_vertical_forward_difference(paddedlevelset)

@test allapprox(Dmx, ones(length(numnodes)))
@test allapprox(Dpx, ones(length(numnodes)))
@test allapprox(Dmy, zeros(length(numnodes)))
@test allapprox(Dpy, zeros(length(numnodes)))
