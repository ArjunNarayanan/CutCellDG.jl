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

function free_slip_bc_nodal_displacement(
    mesh,
    basis,
    cellquads,
    facequads,
    interfacequads,
    stiffness,
    theta0,
    penalty;
    eta = 1,
)

    L, W = CutCellDG.widths(mesh)
    lambda1, mu1 = CutCellDG.lame_coefficients(stiffness, +1)
    transfstress =
        CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)

    sysmatrix = CutCellDG.SystemMatrix()
    sysrhs = CutCellDG.SystemRHS()

    CutCellDG.assemble_displacement_bilinear_forms!(
        sysmatrix,
        basis,
        cellquads,
        stiffness,
        mesh,
    )
    CutCellDG.assemble_bulk_transformation_linear_form!(
        sysrhs,
        transfstress,
        basis,
        cellquads,
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
    CutCellDG.assemble_interelement_transformation_linear_form!(
        sysrhs,
        transfstress,
        basis,
        facequads,
        mesh,
    )

    CutCellDG.assemble_incoherent_interface_condition!(
        sysmatrix,
        basis,
        interfacequads,
        stiffness,
        mesh,
        penalty,
        eta,
    )
    CutCellDG.assemble_incoherent_interface_transformation_linear_form!(
        sysrhs,
        transfstress,
        basis,
        interfacequads,
        mesh,
    )

    CutCellDG.assemble_penalty_displacement_component_bc!(
        sysmatrix,
        sysrhs,
        x->0.0,
        basis,
        facequads,
        stiffness,
        mesh,
        x -> onleftboundary(x, L, W),
        [1.0,0.0],
        penalty,
    )
    CutCellDG.assemble_penalty_displacement_component_transformation_linear_form!(
        sysrhs,
        transfstress,
        basis,
        facequads,
        mesh,
        x -> onleftboundary(x, L, W),
        [1.0,0.0]
    )

    CutCellDG.assemble_penalty_displacement_component_bc!(
        sysmatrix,
        sysrhs,
        x->0.0,
        basis,
        facequads,
        stiffness,
        mesh,
        x -> onbottomboundary(x, L, W),
        [0.0,1.0],
        penalty,
    )
    CutCellDG.assemble_penalty_displacement_component_transformation_linear_form!(
        sysrhs,
        transfstress,
        basis,
        facequads,
        mesh,
        x -> onbottomboundary(x, L, W),
        [0.0,1.0]
    )

    matrix = CutCellDG.sparse_displacement_operator(sysmatrix, mesh)
    rhs = CutCellDG.displacement_rhs_vector(sysrhs, mesh)

    solution = matrix \ rhs

    return solution
end

function construct_mesh_and_quadratures(
    meshwidth,
    nelmts,
    basis,
    corner,
    numqp;
    tinyratio = 0.2,
)
    cgmesh = CutCellDG.CGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basis)
    mesh = CutCellDG.DGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basis)

    levelset = CutCellDG.LevelSet(
        x -> -corner_distance_function(x, corner),
        cgmesh,
        basis,
    )

    cutmesh = CutCellDG.CutMesh(mesh, levelset)
    cellquads = CutCellDG.CellQuadratures(cutmesh, levelset, numqp)
    interfacequads = CutCellDG.InterfaceQuadratures(cutmesh, levelset, numqp)
    facequads = CutCellDG.FaceQuadratures(cutmesh, levelset, numqp)

    mergedwithcell, hasmergedcells = CutCellDG.merge_tiny_cells_in_mesh!(
        cutmesh,
        cellquads,
        facequads,
        interfacequads,
        tinyratio = tinyratio,
    )
    mergedmesh = CutCellDG.MergedMesh(cutmesh, mergedwithcell)

    return mergedmesh, cellquads, facequads, interfacequads
end

function construct_unmerged_mesh_and_quadratures(
    meshwidth,
    nelmts,
    basis,
    interfacecenter,
    interfaceradius,
    numqp,
)
    mesh = CutCellDG.DGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basis)

    levelset = InterpolatingPolynomial(1, basis)
    levelsetcoeffs = CutCellDG.levelset_coefficients(
        x -> -circle_distance_function(x, interfacecenter, interfaceradius),
        mesh,
    )

    cutmesh = CutCellDG.CutMesh(mesh, levelset, levelsetcoeffs)
    cellquads =
        CutCellDG.CellQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
    interfacequads =
        CutCellDG.InterfaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)
    facequads =
        CutCellDG.FaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)

    return cutmesh, cellquads, facequads, interfacequads
end

function symmetric_displacement_gradient(
    celldisp,
    basis,
    point,
    jac,
    vectosymmconverter,
)
    dim = length(vectosymmconverter)
    grad = CutCellDG.transform_gradient(gradient(basis, point), jac)
    NK = sum([
        CutCellDG.make_row_matrix(vectosymmconverter[k], grad[:, k]) for
        k = 1:dim
    ])
    symmdispgrad = NK * celldisp
    return symmdispgrad
end

function parent_strain_at_reference_points(
    nodaldisplacement,
    basis,
    referencepoints,
    referencecellids,
    mesh,
)

    dim = CutCellDG.dimension(mesh)
    dim2, numpts = size(referencepoints)
    @assert dim == dim2
    @assert length(referencecellids) == numpts

    parentstrain = zeros(4, numpts)
    vectosymmconverter = CutCellDG.vector_to_symmetric_matrix_converter()
    jac = CutCellDG.jacobian(mesh)

    for i = 1:numpts
        cellid = referencecellids[i]
        nodeids = CutCellDG.nodal_connectivity(mesh, -1, cellid)
        celldofs = CutCellDG.element_dofs(nodeids, dim)
        celldisp = nodaldisplacement[celldofs]
        point = referencepoints[:, i]

        parentstrain[1:3, i] .= symmetric_displacement_gradient(
            celldisp,
            basis,
            point,
            jac,
            vectosymmconverter,
        )
    end
    return parentstrain
end

function parent_stress(symmdispgrad, stiffness)
    lambda, mu = CutCellDG.lame_coefficients(stiffness, -1)

    inplanestress = stiffness[-1] * symmdispgrad[1:3, :]
    s33 = lambda * (symmdispgrad[1, :] + symmdispgrad[2, :])

    return vcat(inplanestress, s33')
end

function product_strain_at_reference_points(
    nodaldisplacement,
    basis,
    theta0,
    referencepoints,
    referencecellids,
    mesh,
)

    dim = CutCellDG.dimension(mesh)
    dim2, numpts = size(referencepoints)
    @assert dim == dim2
    @assert length(referencecellids) == numpts

    productstrain = zeros(4, numpts)
    vectosymmconverter = CutCellDG.vector_to_symmetric_matrix_converter()
    jac = CutCellDG.jacobian(mesh)

    for i = 1:numpts
        cellid = referencecellids[i]
        nodeids = CutCellDG.nodal_connectivity(mesh, +1, cellid)
        celldofs = CutCellDG.element_dofs(nodeids, dim)
        celldisp = nodaldisplacement[celldofs]
        point = referencepoints[:, i]

        productstrain[1:3, i] .= symmetric_displacement_gradient(
            celldisp,
            basis,
            point,
            jac,
            vectosymmconverter,
        )
    end
    productstrain[[1, 2, 4], :] .-= theta0 / 3
    return productstrain
end

function product_stress(elasticstrain, stiffness, theta0)
    lambda, mu = CutCellDG.lame_coefficients(stiffness, +1)

    inplanestress = stiffness[+1] * elasticstrain[1:3, :]
    inplanecorrection = lambda * elasticstrain[4, :]

    inplanestress[1, :] .+= inplanecorrection
    inplanestress[2, :] .+= inplanecorrection

    s33 =
        (lambda + 2mu) * elasticstrain[4, :] +
        lambda * (elasticstrain[1, :] + elasticstrain[2, :])

    return vcat(inplanestress,s33')
end


function product_stress(
    celldisp,
    basis,
    stiffness,
    transfstress,
    theta0,
    point,
    jac,
    vectosymmconverter,
)

    dim = length(vectosymmconverter)
    lambda, mu = CutCellDG.lame_coefficients(stiffness, +1)

    grad = CutCellDG.transform_gradient(gradient(basis, point), jac)
    NK = sum([
        CutCellDG.make_row_matrix(vectosymmconverter[k], grad[:, k]) for
        k = 1:dim
    ])
    symmdispgrad = NK * celldisp

    inplanestress = (stiffness[+1] * symmdispgrad) - transfstress
    s33 =
        lambda * (symmdispgrad[1] + symmdispgrad[2]) -
        (lambda + 2mu / 3) * theta0

    stress = vcat(inplanestress, s33)

    return stress
end

function product_stress_at_reference_points(
    nodaldisplacement,
    basis,
    stiffness,
    transfstress,
    theta0,
    referencepoints,
    referencecellids,
    mesh,
)

    dim = CutCellDG.dimension(mesh)
    dim2, numpts = size(referencepoints)
    @assert dim == dim2
    @assert length(referencecellids) == numpts

    productstress = zeros(4, numpts)
    vectosymmconverter = CutCellDG.vector_to_symmetric_matrix_converter()
    jac = CutCellDG.jacobian(mesh)

    for i = 1:numpts
        cellid = referencecellids[i]
        nodeids = CutCellDG.nodal_connectivity(mesh, +1, cellid)
        celldofs = CutCellDG.element_dofs(nodeids, dim)
        celldisp = nodaldisplacement[celldofs]
        point = referencepoints[:, i]

        productstress[:, i] .= product_stress(
            celldisp,
            basis,
            stiffness,
            transfstress,
            theta0,
            point,
            jac,
            vectosymmconverter,
        )
    end
    return productstress
end


function parent_stress(
    celldisp,
    basis,
    stiffness,
    point,
    jac,
    vectosymmconverter,
)
    dim = length(vectosymmconverter)
    lambda, mu = CutCellDG.lame_coefficients(stiffness, -1)

    grad = CutCellDG.transform_gradient(gradient(basis, point), jac)
    NK = sum([
        CutCellDG.make_row_matrix(vectosymmconverter[k], grad[:, k]) for
        k = 1:dim
    ])
    symmdispgrad = NK * celldisp

    inplanestress = stiffness[-1] * symmdispgrad
    s33 = lambda * (symmdispgrad[1] + symmdispgrad[2])

    stress = vcat(inplanestress, s33)

    return stress
end

function parent_stress_at_reference_points(
    nodaldisplacement,
    basis,
    stiffness,
    referencepoints,
    referencecellids,
    mesh,
)

    dim = CutCellDG.dimension(mesh)
    dim2, numpts = size(referencepoints)
    @assert dim == dim2
    @assert length(referencecellids) == numpts

    parentstress = zeros(4, numpts)
    vectosymmconverter = CutCellDG.vector_to_symmetric_matrix_converter()
    jac = CutCellDG.jacobian(mesh)

    for i = 1:numpts
        cellid = referencecellids[i]
        nodeids = CutCellDG.nodal_connectivity(mesh, -1, cellid)
        celldofs = CutCellDG.element_dofs(nodeids, dim)
        celldisp = nodaldisplacement[celldofs]
        point = referencepoints[:, i]

        parentstress[:, i] .= parent_stress(
            celldisp,
            basis,
            stiffness,
            point,
            jac,
            vectosymmconverter,
        )
    end
    return parentstress
end

function pressure(stress)
    return -1.0 / 3.0 * (stress[1, :] + stress[2, :] + stress[4, :])
end

function deviatoric_stress(stress, press)
    devstress = copy(stress)
    devstress[1, :] .+= press
    devstress[2, :] .+= press
    devstress[4, :] .+= press
    return devstress
end

function strain_energy(stress,strain)
    product = stress .* strain
    return 0.5*vec(sum(product,dims=1))
end
