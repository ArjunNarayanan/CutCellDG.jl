function assemble_bulk_transformation_linear_form!(
    systemrhs,
    transfstress,
    basis,
    cellquads,
    mesh,
)

    ncells = number_of_cells(mesh)
    dim = dimension(mesh)
    jac = jacobian(mesh)
    detjac = determinant_jacobian(mesh)
    vectosymmconverter = vector_to_symmetric_matrix_converter()

    for cellid = 1:ncells
        cellsign = cell_sign(mesh, cellid)
        check_cellsign(cellsign)
        if cellsign == +1 || cellsign == 0
            pquad = cellquads[+1, cellid]
            rhs = bulk_transformation_rhs(
                transfstress,
                basis,
                pquad,
                dim,
                jac,
                detjac,
                vectosymmconverter,
            )
            nodeids = nodal_connectivity(mesh, +1, cellid)
            edofs = element_dofs(nodeids, dim)
            assemble!(systemrhs, edofs, rhs)
        end
    end
end

function assemble_interelement_transformation_linear_form!(
    systemrhs,
    transfstress,
    basis,
    facequads,
    mesh,
)

    dim = dimension(mesh)
    ncells = number_of_cells(mesh)
    nfaces = number_of_faces_per_cell(facequads)
    faceids = 1:nfaces
    nbrfaceids = [opposite_face(faceid) for faceid in faceids]
    normals = reference_face_normals()
    product_cellsign = +1
    facedetjac = face_determinant_jacobian(mesh)
    vectosymmconverter = vector_to_symmetric_matrix_converter()

    for cellid = 1:ncells
        cellsign = cell_sign(mesh, cellid)
        check_cellsign(cellsign)

        if cellsign == product_cellsign || cellsign == 0
            nodeids1 = nodal_connectivity(mesh, product_cellsign, cellid)

            for faceid in faceids
                nbrcellid = cell_connectivity(mesh, faceid, cellid)
                if cellid < nbrcellid &&
                   solution_cell_id(mesh, product_cellsign, cellid) !=
                   solution_cell_id(mesh, product_cellsign, nbrcellid)

                    nbrcellsign = cell_sign(mesh, nbrcellid)

                    if nbrcellsign == product_cellsign || nbrcellsign == 0
                        quad1 = facequads[product_cellsign, faceid, cellid]
                        quad2 = facequads[
                            product_cellsign,
                            nbrfaceids[faceid],
                            nbrcellid,
                        ]

                        nodeids2 = nodal_connectivity(
                            mesh,
                            product_cellsign,
                            nbrcellid,
                        )

                        assemble_face_interelement_transformation_linear_form!(
                            systemrhs,
                            -transfstress,
                            basis,
                            quad1,
                            normals[faceid],
                            nodeids1,
                            dim,
                            facedetjac[faceid],
                            vectosymmconverter,
                        )
                        assemble_face_interelement_transformation_linear_form!(
                            systemrhs,
                            transfstress,
                            basis,
                            quad2,
                            normals[faceid],
                            nodeids2,
                            dim,
                            facedetjac[nbrfaceids[faceid]],
                            vectosymmconverter,
                        )
                    end
                end
            end
        end
    end
end

function interelement_transformation_linear_form(
    transfstress,
    basis,
    quad,
    normal,
    dim,
    facedetjac,
    vectosymmconverter,
)

    numqp = length(quad)
    normals = repeat(normal, inner = (1, numqp))
    scalearea = repeat([facedetjac], numqp)
    return coherent_interface_transformation_linear_form(
        transfstress,
        basis,
        quad,
        normals,
        dim,
        scalearea,
        vectosymmconverter,
    )
end

function component_transformation_linear_form(
    transfstress,
    basis,
    quad,
    component,
    normal,
    dim,
    facedetjac,
    vectosymmconverter,
)

    numqp = length(quad)
    normals = repeat(normal, inner = (1, numqp))
    components = repeat(component,inner=(1,numqp))
    scalearea = repeat([facedetjac], numqp)
    return component_interface_transformation_linear_form(
        transfstress,
        basis,
        quad,
        components,
        normals,
        dim,
        scalearea,
        vectosymmconverter,
    )
end

function assemble_face_interelement_transformation_linear_form!(
    systemrhs,
    transfstress,
    basis,
    quad,
    normal,
    nodeids,
    dim,
    facedetjac,
    vectosymmconverter,
)

    rhs = interelement_transformation_linear_form(
        transfstress,
        basis,
        quad,
        normal,
        dim,
        facedetjac,
        vectosymmconverter,
    )
    edofs = element_dofs(nodeids, dim)
    assemble!(systemrhs, edofs, rhs)
end

function assemble_face_component_transformation_linear_form!(
    systemrhs,
    transfstress,
    basis,
    quad,
    component,
    normal,
    nodeids,
    dim,
    facedetjac,
    vectosymmconverter,
)

    rhs = component_transformation_linear_form(
        transfstress,
        basis,
        quad,
        component,
        normal,
        dim,
        facedetjac,
        vectosymmconverter,
    )
    edofs = element_dofs(nodeids, dim)
    assemble!(systemrhs, edofs, rhs)
end

function assemble_penalty_displacement_transformation_rhs!(
    sysrhs,
    transfstress,
    basis,
    facequads,
    mesh,
    onboundary,
)

    facemidpoints = reference_face_midpoints()
    normals = reference_face_normals()
    nfaces = number_of_faces_per_cell(facequads)
    facedetjac = face_determinant_jacobian(mesh)
    dim = dimension(mesh)
    ncells = number_of_cells(mesh)
    vectosymmconverter = vector_to_symmetric_matrix_converter()

    for cellid = 1:ncells
        cellsign = cell_sign(mesh, cellid)
        check_cellsign(cellsign)

        if cellsign == +1 || cellsign == 0
            cellmap = cell_map(mesh, +1, cellid)
            for faceid = 1:nfaces
                if cell_connectivity(mesh, faceid, cellid) == 0 &&
                   onboundary(cellmap1(facemidpoints[faceid]))

                    quad = facequads[+1, faceid, cellid]
                    nodeids = nodal_connectivity(mesh, +1, cellid)

                    assemble_face_interelement_transformation_linear_form!(
                        sysrhs,
                        -transfstress,
                        basis,
                        quad,
                        normals[faceid],
                        nodeids,
                        dim,
                        facedetjac[faceid],
                        vectosymmconverter,
                    )
                end
            end
        end
    end
end

function assemble_penalty_displacement_component_transformation_rhs!(
    sysrhs,
    transfstress,
    basis,
    facequads,
    mesh,
    onboundary,
    component,
)

    facemidpoints = reference_face_midpoints()
    normals = reference_face_normals()
    nfaces = number_of_faces_per_cell(facequads)
    facedetjac = face_determinant_jacobian(mesh)
    dim = dimension(mesh)
    ncells = number_of_cells(mesh)
    vectosymmconverter = vector_to_symmetric_matrix_converter()

    for cellid = 1:ncells
        cellsign = cell_sign(mesh, cellid)
        check_cellsign(cellsign)

        if cellsign == +1 || cellsign == 0
            cellmap = cell_map(mesh, +1, cellid)
            for faceid = 1:nfaces
                if cell_connectivity(mesh, faceid, cellid) == 0 &&
                   onboundary(cellmap(facemidpoints[faceid]))

                    quad = facequads[+1, faceid, cellid]
                    nodeids = nodal_connectivity(mesh, +1, cellid)

                    assemble_face_component_transformation_linear_form!(
                        sysrhs,
                        -transfstress,
                        basis,
                        quad,
                        component,
                        normals[faceid],
                        nodeids,
                        dim,
                        facedetjac[faceid],
                        vectosymmconverter,
                    )
                end
            end
        end
    end
end
