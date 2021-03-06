function assemble_interelement_condition!(
    sysmatrix,
    basis,
    facequads,
    stiffness,
    mesh,
    penalty,
    eta,
)

    check_eta(eta)
    uniformquads = uniform_face_quadratures(facequads)
    normals = reference_face_normals()
    facedetjac = face_determinant_jacobian(mesh)
    jac = jacobian(mesh)
    nfaces = number_of_faces_per_cell(facequads)
    dim = dimension(mesh)
    vectosymmconverter = vector_to_symmetric_matrix_converter()

    faceids = 1:nfaces
    nbrfaceids = [opposite_face(faceid) for faceid in faceids]

    uniformtop1 = [
        interelement_traction_operators(
            basis,
            uniformquads[faceid],
            uniformquads[nbrfaceids[faceid]],
            normals[faceid],
            stiffness[+1],
            dim,
            facedetjac[faceid],
            jac,
            vectosymmconverter,
            eta,
        ) for faceid in faceids
    ]
    uniformtop2 = [
        interelement_traction_operators(
            basis,
            uniformquads[faceid],
            uniformquads[nbrfaceids[faceid]],
            normals[faceid],
            stiffness[-1],
            dim,
            facedetjac[faceid],
            jac,
            vectosymmconverter,
            eta,
        ) for faceid in faceids
    ]

    uniformtop = [uniformtop1, uniformtop2]
    uniformmassop = [
        interelement_mass_operators(
            basis,
            uniformquads[faceid],
            uniformquads[nbrfaceids[faceid]],
            dim,
            penalty * facedetjac[faceid],
        ) for faceid in faceids
    ]

    ncells = number_of_cells(mesh)

    for cellid = 1:ncells
        cellsign = cell_sign(mesh, cellid)
        check_cellsign(cellsign)
        if cellsign == +1 || cellsign == -1
            assemble_uniform_cell_interelement_condition!(
                sysmatrix,
                uniformtop,
                uniformmassop,
                mesh,
                cellsign,
                cellid,
                faceids,
            )
        else
            assemble_cut_cell_interelement_condition!(
                sysmatrix,
                basis,
                facequads,
                normals,
                stiffness,
                mesh,
                +1,
                cellid,
                faceids,
                nbrfaceids,
                facedetjac,
                jac,
                vectosymmconverter,
                penalty,
                eta,
            )
            assemble_cut_cell_interelement_condition!(
                sysmatrix,
                basis,
                facequads,
                normals,
                stiffness,
                mesh,
                -1,
                cellid,
                faceids,
                nbrfaceids,
                facedetjac,
                jac,
                vectosymmconverter,
                penalty,
                eta,
            )
        end
    end

end

function interelement_traction_operators(
    basis,
    quad1,
    quad2,
    normal,
    stiffness,
    dim,
    facedetjac,
    jac,
    vectosymmconverter,
    eta,
)
    numqp = length(quad1)
    facescale = repeat([facedetjac], numqp)
    normals = repeat(normal, inner = (1, numqp))
    return coherent_traction_operators(
        basis,
        quad1,
        quad2,
        normals,
        stiffness,
        stiffness,
        dim,
        facescale,
        jac,
        vectosymmconverter,
        eta,
    )
end

function interelement_mass_operators(basis, quad1, quad2, dim, scale)
    numqp = length(quad1)
    facescale = repeat([scale], numqp)
    return coherent_mass_operators(basis, quad1, quad2, dim, facescale)
end

function assemble_uniform_cell_interelement_condition!(
    sysmatrix,
    uniformtop,
    uniformmassop,
    mesh,
    cellsign,
    cellid,
    faceids,
)

    dim = dimension(mesh)
    nodeids1 = nodal_connectivity(mesh, cellsign, cellid)
    row = cell_sign_to_row(cellsign)

    for faceid in faceids
        nbrcellid = cell_connectivity(mesh, faceid, cellid)
        if cellid < nbrcellid &&
           solution_cell_id(mesh, cellsign, nbrcellid) !=
           solution_cell_id(mesh, cellsign, cellid)

            nodeids2 = nodal_connectivity(mesh, cellsign, nbrcellid)
            assemble_interface_condition!(
                sysmatrix,
                nodeids1,
                nodeids2,
                dim,
                uniformtop[row][faceid],
                uniformmassop[faceid],
            )
        end
    end
end

function assemble_cut_cell_interelement_condition!(
    sysmatrix,
    basis,
    facequads,
    normals,
    stiffness,
    mesh,
    cellsign,
    cellid,
    faceids,
    nbrfaceids,
    facedetjac,
    jac,
    vectosymmconverter,
    penalty,
    eta,
)

    dim = dimension(mesh)
    nodeids1 = nodal_connectivity(mesh, cellsign, cellid)
    for faceid in faceids
        nbrcellid = cell_connectivity(mesh, faceid, cellid)
        if cellid < nbrcellid &&
           solution_cell_id(mesh, cellsign, nbrcellid) !=
           solution_cell_id(mesh, cellsign, cellid)

            nbrcellsign = cell_sign(mesh, nbrcellid)
            if nbrcellsign == cellsign || nbrcellsign == 0
                nodeids2 = nodal_connectivity(mesh, cellsign, nbrcellid)
                tractionop = interelement_traction_operators(
                    basis,
                    facequads[cellsign, faceid, cellid],
                    facequads[cellsign, nbrfaceids[faceid], nbrcellid],
                    normals[faceid],
                    stiffness[cellsign],
                    dim,
                    facedetjac[faceid],
                    jac,
                    vectosymmconverter,
                    eta,
                )
                massop = interelement_mass_operators(
                    basis,
                    facequads[cellsign, faceid, cellid],
                    facequads[cellsign, nbrfaceids[faceid], nbrcellid],
                    dim,
                    penalty * facedetjac[faceid],
                )
                assemble_interface_condition!(
                    sysmatrix,
                    nodeids1,
                    nodeids2,
                    dim,
                    tractionop,
                    massop,
                )
            end
        end
    end
end
