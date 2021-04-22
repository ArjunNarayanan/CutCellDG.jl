using Test
using PolynomialBasis
using Revise
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
CutCellDG.load_coefficients!(levelset, 1)
cellmap = CutCellDG.cell_map(mesh, 1)

xq = [0.0, 0.0]
x0 = [0.75, 0.5]
X0 = CutCellDG.inverse(cellmap, x0)
invjac = CutCellDG.inverse_jacobian(mesh)
func = CutCellDG.interpolater(levelset)

grad(X) = CutCellDG.spatial_gradient(func, X, invjac)
hess(X) = CutCellDG.spatial_hessian(func, X, invjac)
invfunc(x) = func(CutCellDG.inverse(cellmap, x))

g0 = grad(X0)
l0 = (xq - x0)' * g0 / (g0' * g0)

p0 = func(X0)
∇p0 = grad(X0)
∇2p0 = hess(X0)
gf = CutCellDG.saye_newton_gradient(x0, l0, xq, p0, ∇p0)
hf = CutCellDG.saye_newton_hessian(l0, ∇p0, ∇2p0)

x1, l1 = CutCellDG.step_saye_newton_iterate(x0, l0, gf, hf, 2.0)
testcp = [0.75, 0.0]
@test allapprox(x1, testcp)

X1 = CutCellDG.inverse(cellmap, x1)
p1 = func(X1)
∇p1 = grad(X1)
∇2p1 = hess(X1)
gf = CutCellDG.saye_newton_gradient(x1, l1, xq, p1, ∇p1)
hf = CutCellDG.saye_newton_hessian(l1, ∇p1, ∇2p1)

x2, l2 = CutCellDG.step_saye_newton_iterate(x1, l1, gf, hf, 2.0)
@test allapprox(x2, testcp)

################################################################################

x0 = [0.75, 0.5]
X0 = CutCellDG.inverse(cellmap, x0)
p0 = func(X0)
∇p0 = grad(X0)
∇2p0 = hess(X0)

x1, l1 = CutCellDG.step_chopp_iterate(x0, xq, p0, ∇p0, 0.5)
@test allapprox(x1, testcp)

X1 = CutCellDG.inverse(cellmap, x1)
p1 = func(X1)
∇p1 = grad(X1)
∇2p0 = hess(X1)

x2, l2 = CutCellDG.step_chopp_iterate(x1, xq, p1, ∇p1, 0.5)
@test allapprox(x2, testcp)



################################################################################
tol = 1e3eps()
x1, flag = CutCellDG.spatial_closest_point(
    xq,
    x0,
    func,
    cellmap,
    tol,
    5.0,
    20,
    1e4eps(),
)
@test allapprox(x1, testcp)
@test flag




################################################################################
mesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], [2, 3], basis)
normal = [1.0, 0.0]
xI = [0.75, 0.0]
levelset =
    CutCellDG.LevelSet(x -> plane_distance_function(x, normal, xI), mesh, basis)
cutmesh = CutCellDG.CutMesh(mesh, levelset)

refseedpoints, seedcellids = CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
seedpoints = CutCellDG.map_to_spatial(refseedpoints, seedcellids, mesh)

querypoints = [
    0.0 0.5 0.8 0.9 1.2 -0.8
    0.1 0.8 0.3 0.5 0.4 0.2
]

dx = CutCellDG.element_size(mesh)
tol = 1e3eps()
boundingradius = 1.5maximum(dx)
closestpoints, closestcellids, flags =
    CutCellDG.closest_points_on_zero_levelset(
        querypoints,
        seedpoints,
        seedcellids,
        levelset,
        tol,
        boundingradius,
    )

testcp = closest_point_on_plane(querypoints, normal, xI)
testcellids = [4,6,4,5,5,4]
@test all(flags)
@test allapprox(testcp,closestpoints)
@test allequal(closestcellids,testcellids)
