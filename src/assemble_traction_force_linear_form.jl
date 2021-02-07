function assemble_traction_force_linear_form!(
    systemrhs,
    tractionfunc,
    basis,
    facequads,
    mesh,
    onboundary,
)

    dim = dimension(mesh)
    facemidpoints = reference_face_midpoints()
    nfaces = number_of_faces_per_cell(facequads)
    facedetjac = face_determinant_jacobian(mesh)
    ncells = number_of_cells(mesh)

    for cellid in 1:ncells
        cellsign = cell_sign(mesh, cellid)
        cellmap1 = cell_map(mesh, +1, cellid)
        cellmap2 = cell_map(mesh, -1, cellid)

        check_cellsign(cellsign)
        for faceid = 1:nfaces
            if cell_connectivity(mesh, faceid, cellid) == 0 && (
                onboundary(cellmap1(facemidpoints[faceid])) ||
                onboundary(cellmap2(facemidpoints[faceid]))
            )
                if cellsign == 0 || cellsign == +1
                    assemble_face_traction_force_linear_form!(
                        systemrhs,
                        tractionfunc,
                        basis,
                        facequads,
                        mesh,
                        +1,
                        faceid,
                        cellid,
                        facedetjac,
                    )
                end
                if cellsign == 0 || cellsign == -1
                    assemble_face_traction_force_linear_form!(
                        systemrhs,
                        tractionfunc,
                        basis,
                        facequads,
                        mesh,
                        -1,
                        faceid,
                        cellid,
                        facedetjac,
                    )
                end
            end
        end
    end
end

function assemble_face_traction_force_linear_form!(
    systemrhs,
    tractionfunc,
    basis,
    facequads,
    mesh,
    cellsign,
    faceid,
    cellid,
    facedetjac,
)
    dim = dimension(mesh)
    cellmap = cell_map(mesh, cellsign, cellid)
    rhs = linear_form(
        tractionfunc,
        basis,
        facequads[cellsign, faceid, cellid],
        cellmap,
        dim,
        facedetjac[faceid],
    )
    nodeids = nodal_connectivity(mesh, cellsign, cellid)
    assemble_cell_rhs!(systemrhs, nodeids, dim, rhs)
end
