using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../useful_routines.jl")
include("plane_incoherent_interface_transformation_solver.jl")
APS = AnalyticalPlaneSolver


L = 1.0
W = 1.0
lambda1, mu1 = 200.0, 80.0
lambda2, mu2 = 150.0, 40.0
penaltyfactor = 1e2
eta = 1
nelmts = 4
dx = L / nelmts
dy = W / nelmts
penalty = penaltyfactor / dx * (lambda1 + mu1 + lambda2 + mu2) * 0.5

theta0 = -0.01

stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)
transfstress =
    CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)

polyorder = 1
numqp = 2
basis = TensorProductBasis(2, polyorder)
dgmesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
cgmesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)

R = 0.51
x0 = [R, 0.0]
normal = [1.0, 0.0]
levelset = CutCellDG.LevelSet(
    x -> plane_distance_function(x, normal, x0),
    cgmesh,
    basis,
)

cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)
facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)
mergedmesh = CutCellDG.MergedMesh!(
    cutmesh,
    cellquads,
    facequads,
    interfacequads,
    tinyratio = 0.3,
)

@assert CutCellDG.has_merged_cells(mergedmesh)

sysmatrix = CutCellDG.SystemMatrix()
sysrhs = CutCellDG.SystemRHS()

CutCellDG.assemble_displacement_bilinear_forms!(
    sysmatrix,
    basis,
    cellquads,
    stiffness,
    mergedmesh,
)
CutCellDG.assemble_bulk_transformation_linear_form!(
    sysrhs,
    transfstress,
    basis,
    cellquads,
    mergedmesh,
)
CutCellDG.assemble_interelement_condition!(
    sysmatrix,
    basis,
    facequads,
    stiffness,
    mergedmesh,
    penalty,
    eta,
)
CutCellDG.assemble_interelement_transformation_linear_form!(
    sysrhs,
    transfstress,
    basis,
    facequads,
    mergedmesh,
)

CutCellDG.assemble_incoherent_interface_condition!(
    sysmatrix,
    basis,
    interfacequads,
    stiffness,
    mergedmesh,
    penalty,
    eta,
)
CutCellDG.assemble_incoherent_interface_transformation_linear_form!(
    sysrhs,
    transfstress,
    basis,
    interfacequads,
    mergedmesh,
)

CutCellDG.assemble_penalty_displacement_component_bc!(
    sysmatrix,
    sysrhs,
    x -> 0.0,
    basis,
    facequads,
    stiffness,
    mergedmesh,
    x -> x[1] ≈ 0.0,
    [1.0, 0.0],
    penalty,
)
CutCellDG.assemble_penalty_displacement_component_transformation_linear_form!(
    sysrhs,
    transfstress,
    basis,
    facequads,
    mergedmesh,
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
    mergedmesh,
    x -> x[2] ≈ 0.0,
    [0.0, 1.0],
    penalty,
)
CutCellDG.assemble_penalty_displacement_component_transformation_linear_form!(
    sysrhs,
    transfstress,
    basis,
    facequads,
    mergedmesh,
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
    mergedmesh,
    x -> x[1] ≈ L,
    [1.0, 0.0],
    penalty,
)
CutCellDG.assemble_penalty_displacement_component_transformation_linear_form!(
    sysrhs,
    transfstress,
    basis,
    facequads,
    mergedmesh,
    x -> x[1] ≈ L,
    [1.0, 0.0],
)

matrix = CutCellDG.sparse_displacement_operator(sysmatrix, mergedmesh)
rhs = CutCellDG.displacement_rhs_vector(sysrhs, mergedmesh)

sol = matrix \ rhs
disp = reshape(sol, 2, :)


solver = APS.PlaneSolver(L, W, R, lambda1, mu1, lambda2, mu2, theta0)


err = mesh_L2_error(
    disp,
    x -> APS.displacement_field(solver, x),
    basis,
    cellquads,
    mergedmesh,
)

@test maximum(err) < 1e2eps()
