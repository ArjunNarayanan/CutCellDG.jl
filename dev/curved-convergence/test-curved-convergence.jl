using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("../../test/useful_routines.jl")

function displacement(alpha, x)
    u1 = alpha * x[2] * sin(pi * x[1])
    u2 = alpha * (x[1]^3 + cos(pi * x[2]))
    return [u1, u2]
end

function body_force(lambda, mu, alpha, x)
    b1 = alpha * (lambda + 2mu) * pi^2 * x[2] * sin(pi * x[1])
    b2 =
        -alpha * (6mu * x[1] + (lambda + mu) * pi * cos(pi * x[1])) +
        alpha * (lambda + 2mu) * pi^2 * cos(pi * x[2])
    return [b1, b2]
end

function onboundary(x, L, W)
    return x[2] ≈ 0.0 || x[1] ≈ L || x[2] ≈ W || x[1] ≈ 0.0
end




xc = [0.5, 0.5]
radius = 0.15
polyorder = 2
penaltyfactor = 1e2
nelmts = 9
numqp = required_quadrature_order(polyorder) + 2
eta = 1

L = 1.0
W = 1.0
lambda, mu = 1.0, 2.0
dx = 1.0 / nelmts
penalty = penaltyfactor / dx * (lambda + mu)
alpha = 0.1
stiffness = CutCellDG.HookeStiffness(lambda, mu, lambda, mu)

basis = TensorProductBasis(2, polyorder)
mesh = CutCellDG.DGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
levelset = InterpolatingPolynomial(1, basis)
levelsetcoeffs = CutCellDG.levelset_coefficients(
    x -> circle_distance_function(x, xc, radius),
    mesh,
)

cutmesh = CutCellDG.CutMesh(mesh, levelset, levelsetcoeffs)
cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
interfacequads =
    CutCellDG.InterfaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)

mergedwithcell, hasmergedcells = CutCellDG.merge_tiny_cells_in_mesh!(
    cutmesh,
    cellquads,
    facequads,
    interfacequads,
)
@assert hasmergedcells
mergedmesh = CutCellDG.MergedMesh(cutmesh, mergedwithcell)
# mergedmesh = cutmesh

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
CutCellDG.assemble_penalty_displacement_bc!(
    sysmatrix,
    sysrhs,
    x -> displacement(alpha, x),
    basis,
    facequads,
    stiffness,
    mergedmesh,
    x -> onboundary(x, L, W),
    penalty,
)
CutCellDG.assemble_body_force!(
    sysrhs,
    x -> body_force(lambda, mu, alpha, x),
    basis,
    cellquads,
    mergedmesh,
)

matrix = CutCellDG.sparse_displacement_operator(sysmatrix, mergedmesh)
rhs = CutCellDG.displacement_rhs_vector(sysrhs, mergedmesh)

solution = matrix \ rhs
nodaldisplacement = reshape(solution, 2, :)

cellerr = cellwise_L2_error(nodaldisplacement,x->displacement(alpha,x),basis,cellquads,mergedmesh)



maxu1errphase1 = sortperm(cellerr[1,1,:],rev=true)
maxu2errphase1 = sortperm(cellerr[2,1,:],rev=true)
maxu1errphase2 = sortperm(cellerr[1,2,:],rev=true)
maxu2errphase2 = sortperm(cellerr[2,2,:],rev=true)


phase1err = cellerr[:,1,maxu1errphase1]
phase2err = cellerr[:,2,maxu1errphase2]

#
# quadareas = CutCellDG.quadrature_areas(cellquads,cutmesh)
# cellid = 32
# cellsign = -1
# row = CutCellDG.cell_sign_to_row(cellsign)
# nbrcellids = CutCellDG.cell_connectivity(mesh,:,cellid)
# nbrcellsign = [nbrid == 0 ? 2 : CutCellDG.cell_sign(cutmesh,nbrid) for nbrid in nbrcellids]
# nbrareas = [nbrid == 0 ? 0.0 : quadareas[2,nbrid] for nbrid in nbrcellids]
#
# largestnbr = nbrcellids[argmax(nbrareas)]
#
# nodeids = CutCellDG.nodal_connectivity(mesh,32)
# update!(levelset,levelsetcoeffs[nodeids])
# using Plots
# xrange = -1:1e-2:1
# contour(xrange,xrange,(x,y)->levelset(x,y),levels=[0.0])


# NOTES: SOLUTION WITHOUT CELL MERGING
# Phase 1 high error cells: 50, 32, 42, 51, 41
# Phase 1 order of magnitude: 2.3e-6
# Phase 2 high error cells: 9, 81, 80, 8, 14, 23, 68, 59
# Phase 2 order of magnitude: 6.6e-7


# NOTES: SOLUTION WITH CELL MERGING
# Phase 1 high error cells: 32, 41, 42, 33
# Phase 1 order of magnitude: 4.5e-4
# Phase 2 high error cells: 22, 24, 30, 23, 21
# Phase 2 order of magnitude: 8.2e-4
