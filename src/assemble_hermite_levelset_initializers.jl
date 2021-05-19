function assemble_hermite_mass_matrix!(
    sysmatrix,
    basis,
    quad,
    mesh,
    dofspernode,
)
    ncells = number_of_cells(mesh)
    detjac = determinant_jacobian(mesh)
    cellmatrixvals = vec(mass_matrix(basis, quad, 1, detjac))

    for cellid = 1:ncells
        nodeids = nodal_connectivity(mesh, cellid)
        assemble_cell_matrix!(sysmatrix, nodeids, dofspernode, cellmatrixvals)
    end
end

function assemble_hermite_linear_form!(
    sysrhs,
    rhsfunc,
    basis,
    quad,
    mesh,
    dofspernode,
)
    numqp = length(quad)
    p, w = points(quad), weights(quad)
    valmatrix = mapslices(basis, p, dims = 1)
    ncells = number_of_cells(mesh)
    detjac = determinant_jacobian(mesh)

    for cellid = 1:ncells
        cellmap = cell_map(mesh, cellid)
        funcvals = detjac * [rhsfunc(cellmap(p[:, i])) * w[i] for i = 1:numqp]
        vals = valmatrix * funcvals
        nodeids = nodal_connectivity(mesh, cellid)
        assemble_cell_rhs!(sysrhs, nodeids, dofspernode, vals)
    end
end
