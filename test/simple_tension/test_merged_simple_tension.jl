using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("../useful_routines.jl")

polyorder = 1
numqp = 2

widths = [2.,1.]
nelmts = [5,5]

basis = TensorProductBasis(2, polyorder)
levelset = InterpolatingPolynomial(1, basis)
mesh = CutCellDG.DGMesh([0.0, 0.0], widths, nelmts, basis)

interfaceangle = 40.
normal = [cosd(interfaceangle), sind(interfaceangle)]
x0 = [1.8, 0.0]
levelsetcoeffs = CutCellDG.levelset_coefficients(
    x -> plane_distance_function(x, normal, x0),
    mesh,
)

cutmesh = CutCellDG.CutMesh(mesh, levelset, levelsetcoeffs)
cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
interfacequads =
    CutCellDG.InterfaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)

mergedwithcell,hasmergedcells = CutCellDG.merge_tiny_cells_in_mesh!(
    cutmesh,
    cellquads,
    facequads,
    interfacequads,
)
@assert hasmergedcells
mergedmesh = CutCellDG.MergedMesh(cutmesh, mergedwithcell)

lambda, mu = (1.0, 2.0)
penalty = 1.0
eta = 1
dx = 0.1
e11 = dx / 2.0
e22 = -lambda / (lambda + 2mu) * e11
dy = e22
s11 = (lambda + 2mu) * e11 + lambda * e22

stiffness = CutCellDG.HookeStiffness(lambda, mu, lambda, mu)

sysmatrix = CutCellDG.SystemMatrix()
sysrhs = CutCellDG.SystemRHS()

CutCellDG.assemble_displacement_bilinear_forms!(
    sysmatrix,
    basis,
    cellquads,
    stiffness,
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
CutCellDG.assemble_coherent_interface_condition!(
    sysmatrix,
    basis,
    interfacequads,
    stiffness,
    mergedmesh,
    penalty,
    eta,
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
CutCellDG.assemble_penalty_displacement_component_bc!(
    sysmatrix,
    sysrhs,
    x -> dx,
    basis,
    facequads,
    stiffness,
    mergedmesh,
    x -> x[1] ≈ widths[1],
    [1.0, 0.0],
    penalty,
)


matrix = CutCellDG.sparse_displacement_operator(sysmatrix, mergedmesh)
rhs = CutCellDG.displacement_rhs_vector(sysrhs,mergedmesh)

solution = matrix\rhs
displacement = reshape(solution,2,:)

function exact_displacement(x,e11,e22)
    return [x[1]*e11,x[2]*e22]
end

err = mesh_L2_error(displacement,x->exact_displacement(x,e11,e22),basis,cellquads,mergedmesh)

@test allapprox(err,zeros(2),1e2eps())
