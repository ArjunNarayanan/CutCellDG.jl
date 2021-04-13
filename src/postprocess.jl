function interpolate_at_reference_points(
    nodalvalues,
    dofspernode,
    basis,
    refpoints,
    refcellids,
    levelsetsign,
    mesh,
)
    dim, numpts = size(refpoints)
    @assert length(refcellids) == numpts

    interpolatedvals = zeros(dofspernode, numpts)

    for idx = 1:numpts
        cellid = refcellids[idx]
        nodeids = nodal_connectivity(mesh, levelsetsign, cellid)
        celldofs = element_dofs(nodeids, dofspernode)
        cellvals = nodalvalues[celldofs]

        vals = basis(refpoints[:, idx])
        NI = interpolation_matrix(vals, dofspernode)

        interpolatedvals[:, idx] = NI * cellvals
    end

    return interpolatedvals
end

function displacement_at_reference_points(
    nodaldisplacement,
    basis,
    refpoints,
    refcellids,
    levelsetsign,
    mesh,
)

    dim = dimension(mesh)
    return interpolate_at_reference_points(
        nodaldisplacement,
        dim,
        basis,
        refpoints,
        refcellids,
        levelsetsign,
        mesh,
    )
end

function plane_strain_stress(
    celldisp,
    basis,
    stiffness,
    referencepoint,
    jac,
    vectosymmconverter,
)

    dim = length(vectosymmconverter)

    grad = transform_gradient(gradient(basis, referencepoint), jac)
    NK = sum([make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim])

    symmdispgrad = NK * celldisp

    inplanestress = stiffness * symmdispgrad

    return inplanestress
end

function stress_at_reference_points(
    nodaldisplacement,
    basis,
    stiffness,
    referencepoints,
    referencecellids,
    levelsetsign,
    mesh,
)

    dim = dimension(mesh)

    nphase, dim2, numpts = size(referencepoints)
    @assert dim2 == dim
    @assert size(referencecellids) == (nphase, numpts)
    @assert nphase == 2

    vectosymmconverter = vector_to_symmetric_matrix_converter()
    jac = jacobian(mesh)

    stress = zeros(3, numpts)
    row = cell_sign_to_row(levelsetsign)

    for i = 1:numpts
        cellid = referencecellids[row, i]
        nodeids = nodal_connectivity(mesh, levelsetsign, cellid)
        celldofs = element_dofs(nodeids, dim)
        celldisp = nodaldisplacement[celldofs]
        referencepoint = referencepoints[row, :, i]

        stress[:, i] .= plane_strain_stress(
            celldisp,
            basis,
            stiffness[levelsetsign],
            referencepoint,
            jac,
            vectosymmconverter,
        )
    end

    return stress
end

function symmetric_displacement_gradient(
    celldisp,
    basis,
    point,
    jac,
    vectosymmconverter,
)
    dim = length(vectosymmconverter)
    grad = transform_gradient(gradient(basis, point), jac)
    NK = sum([make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim])
    symmdispgrad = NK * celldisp
    return symmdispgrad
end

function plane_strain_at_reference_points(
    nodaldisplacement,
    basis,
    referencepoints,
    referencecellids,
    levelsetsign,
    mesh,
)

    @assert levelsetsign == +1 || levelsetsign == -1
    dim = dimension(mesh)
    dim2, numpts = size(referencepoints)
    @assert dim == dim2
    @assert length(referencecellids) == numpts

    strain = zeros(3, numpts)
    vectosymmconverter = vector_to_symmetric_matrix_converter()
    jac = jacobian(mesh)

    for i = 1:numpts
        cellid = referencecellids[i]
        nodeids = nodal_connectivity(mesh, levelsetsign, cellid)
        celldofs = element_dofs(nodeids, dim)
        celldisp = nodaldisplacement[celldofs]
        point = referencepoints[:, i]

        strain[:, i] .= symmetric_displacement_gradient(
            celldisp,
            basis,
            point,
            jac,
            vectosymmconverter,
        )
    end
    return strain
end

function parent_strain(
    nodaldisplacement,
    basis,
    referencepoints,
    referencecellids,
    mesh,
)

    strain = plane_strain_at_reference_points(
        nodaldisplacement,
        basis,
        referencepoints,
        referencecellids,
        -1,
        mesh,
    )
    ndofs, numpts = size(strain)
    return vcat(strain, zeros(numpts)')
end

function product_elastic_strain(
    nodaldisplacement,
    basis,
    theta0,
    referencepoints,
    referencecellids,
    mesh,
)

    strain = plane_strain_at_reference_points(
        nodaldisplacement,
        basis,
        referencepoints,
        referencecellids,
        +1,
        mesh,
    )
    ndofs,numpts = size(strain)
    planestrain = vcat(strain,zeros(numpts)')
    planestrain[[1,2,4],:] .-= theta0/3

    return planestrain
end

function parent_stress(symmdispgrad, stiffness)
    lambda, mu = lame_coefficients(stiffness, -1)

    inplanestress = stiffness[-1] * symmdispgrad[1:3, :]
    s33 = lambda * (symmdispgrad[1, :] + symmdispgrad[2, :])

    return vcat(inplanestress, s33')
end

function product_stress(elasticstrain, stiffness, theta0)
    lambda, mu = lame_coefficients(stiffness, +1)

    inplanestress = stiffness[+1] * elasticstrain[1:3, :]
    inplanecorrection = lambda * elasticstrain[4, :]

    inplanestress[1, :] .+= inplanecorrection
    inplanestress[2, :] .+= inplanecorrection

    s33 =
        (lambda + 2mu) * elasticstrain[4, :] +
        lambda * (elasticstrain[1, :] + elasticstrain[2, :])

    return vcat(inplanestress,s33')
end

function strain_energy(stress,strain)
    product = stress .* strain
    return 0.5*vec(sum(product,dims=1))
end




function traction_force(stressvector, normal)
    return [
        stressvector[1] * normal[1] + stressvector[3] * normal[2],
        stressvector[3] * normal[1] + stressvector[2] * normal[2],
    ]
end

function traction_force_at_points(stresses, normals)
    npts = size(stresses)[2]
    @assert size(normals)[2] == npts

    tractionforce = zeros(2, npts)
    for i = 1:npts
        tractionforce[:, i] .= traction_force(stresses[:, i], normals[:, i])
    end
    return tractionforce
end

function traction_component(traction, normal)
    component = traction .* normal
    return vec(sum(component, dims = 1))
end

function pressure_at_points(stress)
    return -1.0 / 3.0 * (stress[1, :] + stress[2, :] + stress[4, :])
end

function collect_cell_quadratures(cellquads, mesh, cellsign, cellids)
    totalnumqps =
        sum([length(cellquads[cellsign, cellid]) for cellid in cellids])
    dim = dimension(mesh)

    referencepoints = zeros(dim, totalnumqps)
    spatialpoints = zeros(dim, totalnumqps)
    referencecellids = zeros(Int, totalnumqps)

    start = 1
    for cellid in cellids
        quad = cellquads[cellsign, cellid]
        numqps = length(quad)

        stop = start + numqps - 1
        cellmap = cell_map(mesh, cellsign, cellid)

        referencepoints[:, start:stop] = points(quad)
        spatialpoints[:, start:stop] = cellmap(points(quad))
        referencecellids[start:stop] = repeat([cellid], numqps)

        start = stop + 1
    end
    return referencepoints, spatialpoints, referencecellids
end

function collect_cell_quadratures(cellquads, mesh, cellsign)
    ncells = number_of_cells(mesh)
    cellsigns = [cell_sign(mesh, cellid) for cellid = 1:ncells]
    cellids = findall(x -> x == cellsign || x == 0, cellsigns)
    collect_cell_quadratures(cellquads, mesh, cellsign, cellids)
end

function stress_inner_product(stress)
    return (stress[1, :] .* stress[1, :]) +
           (stress[2, :] .* stress[2, :]) +
           (stress[4, :] .* stress[4, :]) +
           2.0 * (stress[3, :] .* stress[3, :])
end

function dilatation(strain)
    return strain[1, :] + strain[2, :] + strain[4, :]
end
