function bulk_modulus(l, m)
    return l + 2m / 3
end

function lame_lambda(k, m)
    return k - 2m / 3
end

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

function onboundary(x, L, W)
    return x[2] ≈ 0.0 || x[1] ≈ L || x[2] ≈ W || x[1] ≈ 0.0
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
    CutCellDG.assemble_penalty_displacement_bc!(
        sysmatrix,
        sysrhs,
        x -> displacement(displacementscale, x),
        basis,
        facequads,
        stiffness,
        mesh,
        x -> onboundary(x, L, W),
        penalty,
    )
    CutCellDG.assemble_body_force!(
        sysrhs,
        x -> body_force(lambda1, mu1, displacementscale, x),
        basis,
        cellquads,
        mesh,
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
