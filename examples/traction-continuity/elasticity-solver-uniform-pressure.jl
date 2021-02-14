function bulk_modulus(l, m)
    return l + 2m / 3
end

function lame_lambda(k, m)
    return k - 2m / 3
end

function stress_field(x)
    return [1.0, 1.0, 0.0]
end

function onleftboundary(x, L, W)
    return x[1] ≈ 0.0
end

function onbottomboundary(x, L, W)
    return x[2] ≈ 0.0
end

function onrightboundary(x, L, W)
    return x[1] ≈ L
end

function ontopboundary(x, L, W)
    return x[2] ≈ W
end

function assemble_linear_system(
    mesh,
    basis,
    cellquads,
    facequads,
    interfacequads,
    stiffness,
    penalty,
    eta,
    displacementscale,
)

    L, W = CutCellDG.widths(mesh)
    lambda1, mu1 = CutCellDG.lame_coefficients(stiffness, +1)

    sysmatrix = CutCellDG.SystemMatrix()
    sysrhs = CutCellDG.SystemRHS()

    CutCellDG.assemble_displacement_bilinear_forms!(
        sysmatrix,
        basis,
        cellquads,
        stiffness,
        mesh,
    )
    CutCellDG.assemble_interelement_condition!(
        sysmatrix,
        basis,
        facequads,
        stiffness,
        mesh,
        penalty,
        eta,
    )
    CutCellDG.assemble_coherent_interface_condition!(
        sysmatrix,
        basis,
        interfacequads,
        stiffness,
        mesh,
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
        mesh,
        x -> onleftboundary(x, L, W),
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
        mesh,
        x -> onbottomboundary(x, L, W),
        [0.0, 1.0],
        penalty,
    )

    CutCellDG.assemble_traction_force_linear_form!(
        sysrhs,
        x -> stress_field(x)[[1, 3]],
        basis,
        facequads,
        mesh,
        x -> onrightboundary(x, L, W),
    )
    CutCellDG.assemble_traction_force_linear_form!(
        sysrhs,
        x -> stress_field(x)[[3, 2]],
        basis,
        facequads,
        mesh,
        x -> ontopboundary(x, L, W),
    )

    matrix = CutCellDG.sparse_displacement_operator(sysmatrix, mesh)
    rhs = CutCellDG.displacement_rhs_vector(sysrhs, mesh)

    return matrix, rhs
end

function nodal_displacement(
    distancefunc,
    stiffness,
    nelmts,
    basis,
    numqp,
    penaltyfactor,
    eta,
)
    L, W = 1.0, 1.0
    lambda1, mu1 = CutCellDG.lame_coefficients(stiffness, +1)
    lambda2, mu2 = CutCellDG.lame_coefficients(stiffness, -1)

    dx = 1.0 / nelmts
    penalty = penaltyfactor / dx * 0.5 * (lambda1 + mu1 + lambda2 + mu2)
    displacementscale = 0.1

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
        tinyratio = 0.2,
    )
    mergedmesh = CutCellDG.MergedMesh(cutmesh, mergedwithcell)

    matrix, rhs = assemble_linear_system(
        mergedmesh,
        basis,
        cellquads,
        facequads,
        interfacequads,
        stiffness,
        penalty,
        eta,
        displacementscale,
    )

    nodaldisplacement = matrix \ rhs

    return nodaldisplacement, mergedmesh, cellquads, facequads, interfacequads
end
