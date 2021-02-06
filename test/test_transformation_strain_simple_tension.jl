using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("useful_routines.jl")



L = 1.0
W = 1.0
lambda1, mu1 = 1.0, 2.0
K1 = (lambda1 + 2mu1 / 3)
lambda2, mu2 = 2.0, 4.0
theta0 = -0.067
e22 = K1 * theta0 / (lambda1 + 2mu1)
penaltyfactor = 1e2
nelmts = 1
dx = 1.0 / nelmts
penalty = penaltyfactor / dx * (lambda1 + mu1)
stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)
transfstress =
    CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)


polyorder = 1
numqp = required_quadrature_order(polyorder) + 2

basis = TensorProductBasis(2, polyorder)
mesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
levelset = InterpolatingPolynomial(1, basis)
levelsetcoeffs = ones(CutCellDG.number_of_nodes(mesh))

cutmesh = CutCellDG.CutMesh(mesh, levelset, levelsetcoeffs)
cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
interfacequads =
    CutCellDG.InterfaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)

sysmatrix = CutCellDG.SystemMatrix()
sysrhs = CutCellDG.SystemRHS()

CutCellDG.assemble_displacement_bilinear_forms!(
    sysmatrix,
    basis,
    cellquads,
    stiffness,
    cutmesh,
)
CutCellDG.assemble_bulk_transformation_linear_form!(
    sysrhs,
    transfstress,
    basis,
    cellquads,
    cutmesh,
)

CutCellDG.assemble_penalty_displacement_component_bc!(
    sysmatrix,
    sysrhs,
    x -> 0.0,
    basis,
    facequads,
    stiffness,
    cutmesh,
    x -> x[1] ≈ 0.0,
    [1.0, 0.0],
    penalty,
)
CutCellDG.assemble_penalty_displacement_component_transformation_rhs!(
    sysrhs,
    transfstress,
    basis,
    facequads,
    cutmesh,
    x -> x[1] ≈ 0.0,
    [1.0, 0.0],
)

CutCellDG.assemble_penalty_displacement_component_bc!(
    sysmatrix,
    sysrhs,
    x -> 0.0,
    basis,
    facequads,
    stiffness,
    cutmesh,
    x -> x[2] ≈ 0.0,
    [0.0, 1.0],
    penalty,
)
CutCellDG.assemble_penalty_displacement_component_transformation_rhs!(
    sysrhs,
    transfstress,
    basis,
    facequads,
    cutmesh,
    x -> x[2] ≈ 0.0,
    [0.0, 1.0],
)

CutCellDG.assemble_penalty_displacement_component_bc!(
    sysmatrix,
    sysrhs,
    x -> 0.0,
    basis,
    facequads,
    stiffness,
    cutmesh,
    x -> x[1] ≈ L,
    [1.0, 0.0],
    penalty,
)
CutCellDG.assemble_penalty_displacement_component_transformation_rhs!(
    sysrhs,
    transfstress,
    basis,
    facequads,
    cutmesh,
    x -> x[1] ≈ L,
    [1.0, 0.0],
)


matrix = CutCellDG.sparse_displacement_operator(sysmatrix, cutmesh)
rhs = CutCellDG.displacement_rhs_vector(sysrhs, cutmesh)

sol = matrix \ rhs
disp = reshape(sol, 2, :)

testdisp = [
    0.0 0.0 0.0 0.0
    0.0 e22 0.0 e22
]
@test allapprox(disp, testdisp, 1e2eps())
