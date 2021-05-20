using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("../useful_routines.jl")

polyorder = 1
numqp = 2

basis = LagrangeTensorProductBasis(2, polyorder)
x0 = [0.0, 0.0]
widths = [3.0, 1.0]
nelements = [3, 1]
dim, nodesperelement = size(interpolation_points(basis))
mesh = CutCellDG.CGMesh([0.0, 0.0], [3.0, 1.0], [3, 1], nodesperelement)
normal = [1.0, 0.0]
x0 = [0.5, 0.0]
levelset =
    CutCellDG.LevelSet(x -> plane_distance_function(x, normal, x0), mesh, basis)
nodalcoordinates = CutCellDG.nodal_coordinates(mesh)

tol = 1e-3
perturbation = 0.0

cellsign = CutCellDG.cell_sign!(levelset, tol, perturbation)
@test allequal(cellsign, [0, 1, 1])

normal = [1.0, 0.0]
x0 = [1.0, 0.0]
tol = 1e-3
perturbation = 1e-2
levelset =
    CutCellDG.LevelSet(x -> plane_distance_function(x, normal, x0), mesh, basis)
cellsign = CutCellDG.cell_sign!(levelset, tol, perturbation)

@test allapprox(
    CutCellDG.coefficients(levelset, 1),
    [-1.0 + perturbation, -1.0 + perturbation, perturbation, perturbation],
)
@test allapprox(
    CutCellDG.coefficients(levelset, 2),
    [perturbation, perturbation, 1.0, 1.0],
)
@test allapprox(CutCellDG.coefficients(levelset, 3), [1.0, 1.0, 2.0, 2.0])


basis = HermiteTensorProductBasis(2)
quad = tensor_product_quadrature(2, 4)
normal = [1.0, 0.0]
x0 = [1.0, 0.0]
tol = 1e-3
perturbation = 1e-2
levelset = CutCellDG.LevelSet(
    x -> plane_distance_function(x, normal, x0),
    mesh,
    basis,
    quad,
)
cellsign = CutCellDG.cell_sign!(levelset,tol,perturbation)
testcellsign = [0,1,1]
@test allequal(testcellsign,cellsign)
coeffs1 = reshape(CutCellDG.coefficients(levelset,1),4,:)
jac = CutCellDG.jacobian(mesh)
testcoeffs1 = [-1+perturbation   -1+perturbation   perturbation   perturbation
                0.0               0.0              0.0            0.0
                jac[1]            jac[1]           jac[1]         jac[1]
                0.0               0.0              0.0            0.0]
@test allapprox(coeffs1,testcoeffs1,1e3eps())
coeffs2 = reshape(CutCellDG.coefficients(levelset,2),4,:)
testcoeffs2 = [perturbation    perturbation   1.0            1.0
               0.0             0.0            0.0            0.0
               jac[1]          jac[1]         jac[1]         jac[1]
               0.0             0.0            0.0            0.0]
@test allapprox(coeffs2,testcoeffs2,1e4eps())
coeffs3 = reshape(CutCellDG.coefficients(levelset,3),4,:)
testcoeffs3 = [1.0     1.0    2.0    2.0
               0.0     0.0    0.0    0.0
               jac[1]  jac[1] jac[1] jac[1]
               0.0     0.0    0.0    0.0]
@test allapprox(coeffs3,testcoeffs3,1e4eps())

basis = LagrangeTensorProductBasis(2,1)
refpoints = interpolation_points(basis)
dgmesh = CutCellDG.DGMesh(x0,widths,nelements,refpoints)
posactivenodeids = CutCellDG.active_node_ids(dgmesh,+1,cellsign)
@test allequal(posactivenodeids, 1:12)
negactivenodeids = CutCellDG.active_node_ids(dgmesh,-1, cellsign)
@test allequal(negactivenodeids, 1:4)

totalnumnodes = CutCellDG.number_of_nodes(dgmesh)
cutmeshnodeids =
    CutCellDG.cut_mesh_nodeids(posactivenodeids, negactivenodeids, totalnumnodes)
testcutmeshnodeids = [
    1 2 3 4 5 6 7 8 9 10 11 12
    13 14 15 16 0 0 0 0 0 0 0 0
]
@test allequal(cutmeshnodeids, testcutmeshnodeids)


cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
@test allequal(CutCellDG.nodal_connectivity(cutmesh, +1, 1), [1, 2, 3, 4])
@test allequal(CutCellDG.nodal_connectivity(cutmesh, -1, 1), [13, 14, 15, 16])
