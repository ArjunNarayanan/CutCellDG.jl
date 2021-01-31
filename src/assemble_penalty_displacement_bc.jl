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
    return surface_traction_operator(
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
                d,
                jac,
                vectosymmconverter,
            ),
        ) for (q, n, d) in zip(uniformquads, normals, facedetjac)
    ]

    for cellid in cellids
        cellmap = cell_map(mesh, cellid)
        cellsign = cell_sign(mesh, cellid)
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

function assemble_boundary_mass_operator!()

end
