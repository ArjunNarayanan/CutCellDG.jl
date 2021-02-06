using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCellDG
include("useful_routines.jl")

function displacement(alpha, x)
    u1 = alpha * x[2] * sin(pi * x[1])
    u2 = alpha * (x[1]^3 + cos(pi * x[2]))
    return [u1, u2]
end

function stress_field(lambda, mu, alpha, x)
    s11 =
        (lambda + 2mu) * alpha * pi * x[2] * cos(pi * x[1]) -
        lambda * alpha * pi * sin(pi * x[2])
    s22 =
        -(lambda + 2mu) * alpha * pi * sin(pi * x[2]) +
        lambda * alpha * pi * x[2] * cos(pi * x[1])
    s12 = alpha * mu * (3x[1]^2 + sin(pi * x[1]))
    return [s11, s22, s12]
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

function compute_stress_error(distancefunc, nelmts, polyorder, numqp, penaltyfactor; eta = +1)
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
    levelsetcoeffs = CutCellDG.levelset_coefficients(distancefunc, mesh)

    cutmesh = CutCellDG.CutMesh(mesh, levelset, levelsetcoeffs)
    cellquads =
        CutCellDG.CellQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
    interfacequads =
        CutCellDG.InterfaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
    facequads =
        CutCellDG.FaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)

    mergedwithcell, hasmergedcells = CutCellDG.merge_tiny_cells_in_mesh!(
        cutmesh,
        cellquads,
        facequads,
        interfacequads,
    )
    @assert hasmergedcells
    mergedmesh = CutCellDG.MergedMesh(cutmesh, mergedwithcell)

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

    nodaldisplacement = matrix \ rhs

    stresserr = stress_L2_error(
        nodaldisplacement,
        basis,
        cellquads,
        stiffness,
        mergedmesh,
        x -> stress_field(lambda, mu, alpha, x),
    )
    return stresserr
end

function update_cell_stress_error!(
    err,
    basis,
    stiffness,
    celldisp,
    quad,
    jac,
    detjac,
    cellmap,
    vectosymmconverter,
    exactsolution,
)

    dim = length(vectosymmconverter)
    for (p, w) in quad
        grad = CutCellDG.transform_gradient(gradient(basis, p), jac)
        NK = sum([
            CutCellDG.make_row_matrix(vectosymmconverter[k], grad[:, k]) for
            k = 1:dim
        ])
        symmdispgrad = NK * celldisp

        numericalstress = stiffness * symmdispgrad
        exactstress = exactsolution(cellmap(p))

        err .+= (numericalstress - exactstress) .^ 2 * detjac * w
    end
end

function stress_L2_error(
    nodaldisplacement,
    basis,
    cellquads,
    stiffness,
    mesh,
    exactsolution,
)

    err = zeros(3)
    numcells = CutCellDG.number_of_cells(mesh)

    dim = CutCellDG.dimension(mesh)
    jac = CutCellDG.jacobian(mesh)
    detjac = CutCellDG.determinant_jacobian(mesh)
    vectosymmconverter = CutCellDG.vector_to_symmetric_matrix_converter()

    for cellid = 1:numcells
        cellsign = CutCellDG.cell_sign(mesh, cellid)

        CutCellDG.check_cellsign(cellsign)
        if cellsign == -1 || cellsign == 0
            nodeids = CutCellDG.nodal_connectivity(mesh, -1, cellid)
            celldofs = CutCellDG.element_dofs(nodeids, dim)
            celldisp = nodaldisplacement[celldofs]
            quad = cellquads[-1, cellid]
            cellmap = CutCellDG.cell_map(mesh,-1,cellid)

            update_cell_stress_error!(
                err,
                basis,
                stiffness[-1],
                celldisp,
                quad,
                jac,
                detjac,
                cellmap,
                vectosymmconverter,
                exactsolution,
            )
        end

        if cellsign == +1 || cellsign == 0
            nodeids = CutCellDG.nodal_connectivity(mesh, +1, cellid)
            celldofs = CutCellDG.element_dofs(nodeids, dim)
            celldisp = nodaldisplacement[celldofs]
            quad = cellquads[+1, cellid]
            cellmap = CutCellDG.cell_map(mesh,+1,cellid)

            update_cell_stress_error!(
                err,
                basis,
                stiffness[+1],
                celldisp,
                quad,
                jac,
                detjac,
                cellmap,
                vectosymmconverter,
                exactsolution,
            )
        end
    end
    return sqrt.(err)
end




powers = [3, 4, 5]
nelmts = [2^p + 1 for p in powers]
dx = 1.0 ./ nelmts
interface_center = [1.0, 0.5]
interface_radius = 0.45
polyorder = 2
numqp = required_quadrature_order(polyorder) + 2
penaltyfactor = 1e3
err = [
    compute_stress_error(
        x -> circle_distance_function(x, interface_center, interface_radius),
        ne,
        polyorder,
        numqp,
        penaltyfactor,
    ) for ne in nelmts
]
serr = [[er[i] for er in err] for i = 1:3]
rates = [convergence_rate(dx, v) for v in serr]
@test all([all(rates[i] .> 1.95) for i = 1:3])





powers = [3, 4, 5]
nelmts = [2^p + 1 for p in powers]
dx = 1.0 ./ nelmts
interface_center = [0.3, 0.8]
interface_radius = 0.15
polyorder = 2
numqp = required_quadrature_order(polyorder) + 2
penaltyfactor = 1e2
err = [
    compute_stress_error(
        x -> circle_distance_function(x, interface_center, interface_radius),
        ne,
        polyorder,
        numqp,
        penaltyfactor,
    ) for ne in nelmts
]
serr = [[er[i] for er in err] for i = 1:3]
rates = [convergence_rate(dx, v) for v in serr]
@test all([all(rates[i] .> 1.95) for i = 1:3])
