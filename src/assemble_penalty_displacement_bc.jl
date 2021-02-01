function boundary_traction_operator(
    basis,
    quad,
    normal,
    stiffness,
    dim,
    facedetjac,
    jac,
    vectosymmconverter,
)

    numqp = length(quad)
    normals = repeat(normal, inner = (1, numqp))
    scalearea = repeat([facedetjac], numqp)
    return -1.0*urface_traction_operator(
        basis,
        quad,
        quad,
        normals,
        stiffness,
        dim,
        scalearea,
        jac,
        vectosymmconverter,
    )
end

function assemble_boundary_traction_operator!(
    sysmatrix,
    basis,
    facequads,
    stiffness,
    mesh,
    onboundary,
)

    facemidpoints = reference_face_midpoints()
    nfaces = number_of_faces_per_cell(facequads)
    facedetjac = face_determinant_jacobian(mesh)
    jac = jacobian(mesh)
    dim = dimension(mesh)
    vectosymmconverter = vector_to_symmetric_matrix_converter()
    normals = reference_face_normals()
    uniformquads = uniform_face_quadratures(facequads)

    faceids = 1:nfaces

    uniformtop1 = [
        vec(
            boundary_traction_operator(
                basis,
                q,
                n,
                stiffness[+1],
                dim,
                d,
                jac,
                vectosymmconverter,
            ),
        ) for (q, n, d) in zip(uniformquads, normals, facedetjac)
    ]
    uniformtop1 = [
        vec(
            boundary_traction_operator(
                basis,
                q,
                n,
                stiffness[-1],
                dim,
                d,
                jac,
                vectosymmconverter,
            ),
        ) for (q, n, d) in zip(uniformquads, normals, facedetjac)
    ]

    ncells = number_of_cells(mesh)

    for cellid = 1:ncells
        cellsign = cell_sign(mesh, cellid)
        cellmap = cell_map(mesh, cellsign, cellid)
        check_cellsign(cellsign)
        for faceid in faceids
            if cell_connectivity(mesh, faceid, cellid) == 0 &&
               onboundary(cellmap(facemidpoints[faceid]))

                if cellsign == +1
                    nodeids = nodal_connectivity(mesh, +1, cellid)
                    assemble_cell_matrix!(
                        sysmatrix,
                        nodeids,
                        dim,
                        uniformtop1[faceid],
                    )
                elseif cellsign == -1
                    nodeids = nodal_connectivity(mesh, -1, cellid)
                    assemble_cell_matrix!(
                        sysmatrix,
                        nodeids,
                        dim,
                        uniformtop2[faceid],
                    )
                else
                    assemble_cut_cell_boundary_traction_operator!(
                        sysmatrix,
                        basis,
                        facequads,
                        normals,
                        stiffness,
                        mesh,
                        +1,
                        faceid,
                        cellid,
                        facedetjac,
                        jac,
                        vectosymmconverter,
                    )
                    assemble_cut_cell_boundary_traction_operator!(
                        sysmatrix,
                        basis,
                        facequads,
                        normals,
                        stiffness,
                        mesh,
                        -1,
                        faceid,
                        cellid,
                        facedetjac,
                        jac,
                        vectosymmconverter,
                    )
                end

            end
        end
    end
end

function assemble_cut_cell_boundary_traction_operator!(
    sysmatrix,
    basis,
    facequads,
    normals,
    stiffness,
    mesh,
    cellsign,
    faceid,
    cellid,
    facedetjac,
    jac,
    vectosymmconverter,
)

    dim = dimension(mesh)
    operator = vec(
        boundary_traction_operator(
            basis,
            facequads[cellsign, faceid, cellid],
            normals[faceid],
            stiffness[cellsign],
            dim,
            detjac[faceid],
            jac,
            vectosymmconverter,
        ),
    )
    nodeids = nodal_connectivity(mesh, cellsign, cellid)
    assemble_cell_matrix!(sysmatrix, nodeids, dim, operator)
end

function boundary_mass_operator(basis, quad, dim, scale)
    numqp = length(quad)
    scalearea = repeat([scale], numqp)
    return mass_matrix(basis, quad, quad, dim, scalearea)
end

function assemble_boundary_mass_operator!(
    sysmatrix,
    basis,
    facequads,
    mesh,
    onboundary,
    penalty,
)

    uniformquads = uniform_face_quadratures(facequads)
    facedetjac = face_determinant_jacobian(mesh)
    nfaces = number_of_faces_per_cell(facequads)
    dim = dimension(mesh)
    facemidpoints = reference_face_midpoints()

    faceids = 1:nfaces

    uniformmassops = [
        vec(boundary_mass_operator(basis, q, dim, penalty * d)) for
        (q, d) in zip(uniformquads, facedetjac)
    ]

    ncells = number_of_cells(mesh)

    for cellid = 1:ncells
        cellsign = cell_sign(mesh, cellid)
        cellmap = cell_map(mesh, cellsign, cellid)
        check_cellsign(cellsign)
        for faceid in faceids
            if cell_connectivity(mesh, faceid, cellid) == 0 &&
               onboundary(cellmap(facemidpoints[faceid]))

                if cellsign == +1 || cellsign == -1
                    nodeids = nodal_connectivity(mesh, cellsign, cellid)
                    assemble_cell_matrix!(
                        sysmatrix,
                        nodeids,
                        dim,
                        uniformmassops[faceid],
                    )
                else
                    assemble_cut_cell_boundary_mass_operator!(
                        sysmatrix,
                        basis,
                        facequads,
                        mesh,
                        +1,
                        faceid,
                        cellid,
                        facedetjac,
                        penalty,
                    )
                    assemble_cut_cell_boundary_mass_operator!(
                        sysmatrix,
                        basis,
                        facequads,
                        mesh,
                        -1,
                        faceid,
                        cellid,
                        facedetjac,
                        penalty,
                    )
                end
            end
        end
    end
end

function assemble_cut_cell_boundary_mass_operator!(
    sysmatrix,
    basis,
    facequads,
    mesh,
    cellsign,
    faceid,
    cellid,
    facedetjac,
    penalty,
)

    dim = dimension(mesh)
    operator = vec(
        boundary_mass_operator(
            basis,
            facequads[cellsign, faceid, cellid],
            dim,
            penalty * facedetjac[faceid],
        ),
    )
    nodeids = nodal_connectivity(mesh, cellsign, cellid)
    assemble_cell_matrix!(sysmatrix, nodeids, dim, operator)
end

function assemble_boundary_displacement_rhs!(
    sysrhs,
    bcfunc,
    basis,
    facequads,
    mesh,
    onboundary,
    penalty,
)

    facemidpoints = reference_face_midpoints()
    nfaces = number_of_faces_per_cell(facequads)
    facedetjac = face_determinant_jacobian(mesh)
    ncells = number_of_cells(mesh)
    faceids = 1:nfaces

    for cellid = 1:ncells
        cellsign = cell_sign(mesh, cellid)
        cellmap = cell_map(mesh, cellsign, cellid)
        check_cellsign(cellsign)
        for faceid in faceids
            if cell_connectivity(mesh, faceid, cellid) == 0 &&
               onboundary(cellmap(facemidpoints[faceid]))
                if cellsign == +1 || cellsign == 0
                    assemble_face_boundary_displacement_rhs!(
                        sysrhs,
                        bcfunc,
                        basis,
                        facequads,
                        mesh,
                        +1,
                        faceid,
                        cellid,
                        facedetjac,
                        penalty,
                    )
                end
                if cellsign == -1 || cellsign == 0
                    assemble_face_boundary_displacement_rhs!(
                        sysrhs,
                        bcfunc,
                        basis,
                        facequads,
                        mesh,
                        -1,
                        faceid,
                        cellid,
                        facedetjac,
                        penalty,
                    )
                end
            end
        end
    end
end

function assemble_face_boundary_displacement_rhs!(
    sysrhs,
    bcfunc,
    basis,
    facequads,
    mesh,
    cellsign,
    faceid,
    cellid,
    facedetjac,
    penalty,
)

    dim = dimension(mesh)
    cellmap = cell_map(mesh, cellsign, cellid)
    rhsvals =
        penalty * linear_form(
            bcfunc,
            basis,
            facequads[cellsign, faceid, cellid],
            cellmap,
            dim,
            facedetjac[faceid],
        )
    nodeids = nodal_connectivity(mesh, cellsign, cellid)
    assemble_cell_rhs!(sysrhs, nodeids, dim, rhsvals)
end

function assemble_penalty_displacement_bc!(
    sysmatrix,
    sysrhs,
    bcfunc,
    basis,
    facequads,
    stiffness,
    mesh,
    onboundary,
    penalty,
)

    assemble_boundary_traction_operator!(
        sysmatrix,
        basis,
        facequads,
        stiffness,
        mesh,
        onboundary,
    )
    assemble_boundary_mass_operator!(
        sysmatrix,
        basis,
        facequads,
        mesh,
        onboundary,
        penalty,
    )
    assemble_boundary_displacement_rhs!(
        sysrhs,
        bcfunc,
        basis,
        facequads,
        mesh,
        onboundary,
        penalty,
    )
end
