function assemble_cell_body_force!(
    systemrhs,
    rhsfunc,
    basis,
    cellquads,
    mesh,
    cellsign,
    cellid,
)

    dim = dimension(mesh)
    detjac = determinant_jacobian(mesh)
    cellmap = cell_map(mesh, cellsign, cellid)
    quad = cellquads[cellsign, cellid]
    rhs = linear_form(rhsfunc, basis, quad, cellmap, dim, detjac)
    nodeids = nodal_connectivity(mesh, cellsign, cellid)
    assemble_cell_rhs!(systemrhs, nodeids, dim, rhs)
end

function assemble_body_force!(systemrhs, rhsfunc, basis, cellquads, mesh)

    ncells = number_of_cells(mesh)
    for cellid = 1:ncells
        cellsign = cell_sign(mesh, cellid)
        check_cellsign(cellsign)

        if cellsign == +1 || cellsign == 0
            assemble_cell_body_force!(
                systemrhs,
                rhsfunc,
                basis,
                cellquads,
                mesh,
                +1,
                cellid,
            )
        end
        if cellsign == -1 || cellsign == 0
            assemble_cell_body_force!(
                systemrhs,
                rhsfunc,
                basis,
                cellquads,
                mesh,
                -1,
                cellid,
            )
        end
    end
end
