function assemble_cut_cell_displacement_bilinear_form!(
    sysmatrix,
    basis,
    cellquads,
    stiffness,
    mesh,
    cellsign,
    cellid,
)

    dim = dimension(mesh)
    jac = jacobian(mesh)
    quad = cellquads[cellsign, cellid]
    cellmatrix =
        vec(displacement_bilinear_form(basis, quad, stiffness[cellsign], jac, dim))
    nodeids = nodal_connectivity(mesh, cellsign, cellid)
    assemble_cell_matrix!(sysmatrix, nodeids, dim, cellmatrix)
end

function assemble_displacement_bilinear_forms!(
    sysmatrix,
    basis,
    cellquads,
    stiffness,
    mesh,
)

    ncells = number_of_cells(mesh)
    jac = jacobian(mesh)
    dim = dimension(mesh)

    uniformquad = uniform_cell_quadrature(cellquads)
    uniformbf1 =
        vec(displacement_bilinear_form(basis, uniformquad, stiffness[+1], jac, dim))
    uniformbf2 =
        vec(displacement_bilinear_form(basis, uniformquad, stiffness[-1], jac, dim))

    uniformcellmatrices = [uniformbf1, uniformbf2]

    for cellid = 1:ncells
        s = cell_sign(mesh, cellid)
        @assert s == +1 || s == 0 || s == -1
        if s == +1
            nodeids = nodal_connectivity(mesh, +1, cellid)
            assemble_cell_matrix!(sysmatrix, nodeids, dim, uniformbf1)
        elseif s == -1
            nodeids = nodal_connectivity(mesh, -1, cellid)
            assemble_cell_matrix!(sysmatrix, nodeids, dim, uniformbf2)
        else
            assemble_cut_cell_displacement_bilinear_form!(
                sysmatrix,
                basis,
                cellquads,
                stiffness,
                mesh,
                +1,
                cellid,
            )
            assemble_cut_cell_displacement_bilinear_form!(
                sysmatrix,
                basis,
                cellquads,
                stiffness,
                mesh,
                -1,
                cellid,
            )
        end
    end
end
