struct InterElementTractionOperatorValues
    nn::Any
    np::Any
    pn::Any
    pp::Any
    nnT::Any
    npT::Any
    pnT::Any
    ppT::Any
    function InterElementTractionOperatorValues(nntop, nptop, pntop, pptop, eta)
        nn = -0.5 * vec(nntop)
        np = -0.5 * vec(nptop)
        pn = -0.5 * vec(pntop)
        pp = -0.5 * vec(pptop)

        nnT = -0.5 * eta * vec(transpose(nntop))
        npT = -0.5 * eta * vec(transpose(nptop))
        pnT = -0.5 * eta * vec(transpose(pntop))
        ppT = -0.5 * eta * vec(transpose(pptop))

        new(nn, np, pn, pp, nnT, npT, pnT, ppT)
    end
end

function interelement_traction_operators(
    basis,
    quad1,
    quad2,
    normal,
    stiffness,
    facedetjac,
    jac,
    eta,
)
    nntop = face_traction_operator(
        basis,
        quad1,
        quad1,
        normal,
        stiffness,
        facedetjac,
        jac,
    )
    nptop = face_traction_operator(
        basis,
        quad1,
        quad2,
        normal,
        stiffness,
        facedetjac,
        jac,
    )
    pntop = face_traction_operator(
        basis,
        quad2,
        quad1,
        normal,
        stiffness,
        facedetjac,
        jac,
    )
    pptop = face_traction_operator(
        basis,
        quad2,
        quad2,
        normal,
        stiffness,
        facedetjac,
        jac,
    )

    return InterElementTractionOperatorValues(nntop, nptop, pntop, pptop, eta)
end

struct InterElementMassOperatorValues
    nn::Any
    np::Any
    pn::Any
    pp::Any
    function InterElementMassOperatorValues(nnmop, npmop, pnmop, ppmop)
        nn = vec(nnmop)
        np = vec(npmop)
        pn = vec(pnmop)
        pp = vec(ppmop)

        new(nn, np, pn, pp)
    end
end

function interelement_mass_operators(basis, quad1, quad2, facescale)
    dim = dimension(basis)
    nn = mass_matrix(basis, quad1, quad1, facescale, dim)
    np = mass_matrix(basis, quad1, quad2, facescale, dim)
    pn = mass_matrix(basis, quad2, quad1, facescale, dim)
    pp = mass_matrix(basis, quad2, quad2, facescale, dim)

    return InterElementMassOperatorValues(nn, np, pn, pp)
end

function assemble_interelement_condition!(
    sysmatrix,
    basis,
    facequads,
    stiffness,
    mesh,
    penalty,
    eta,
)

    @assert eta == 1 || eta == 0 || eta == -1
    uniformquads = uniform_face_quadratures(facequads)
    normals = reference_face_normals()
    facedetjac = face_determinant_jacobian(mesh)
    jac = jacobian(mesh)
    nfaces = number_of_faces_per_cell(facequads)
    dim = dimension(mesh)

    faceids = 1:nfaces
    nbrfaceids = [opposite_face(faceid) for faceid in faceids]

    uniformtop1 = [
        interelement_traction_operators(
            basis,
            uniformquads[faceid],
            uniformquads[nbrfaceids[faceid]],
            normals[faceid],
            stiffness[+1],
            facedetjac[faceid],
            jac,
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
            facedetjac[faceid],
            jac,
            eta,
        ) for faceid in faceids
    ]

    uniformtop = [uniformtop1, uniformtop2]
    uniformmassop = [
        interelement_mass_operators(
            basis,
            uniformquads[faceid],
            uniformquads[nbrfaceids[faceid]],
            penalty * facedetjac[faceid],
        ) for faceid in faceids
    ]

    ncells = number_of_cells(mesh)

    for cellid = 1:ncells
        cellsign = cell_sign(mesh, cellid)
        @assert cellsign == +1 || cellsign == -1
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
                facequads,
                normals,
                stiffness,
                mesh,
                +1,
                cellid,
                faceids,
                facedetjac,
                jac,
                penalty,
                eta,
            )
            assemble_cut_cell_interelement_condition!(
                sysmatrix,
                facequads,
                normals,
                stiffness,
                mesh,
                -1,
                cellid,
                faceids,
                facedetjac,
                jac,
                penalty,
                eta,
            )
        end
    end

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
            assemble_face_interelement_condition!(
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
    facequads,
    normals,
    stiffness,
    mesh,
    cellsign,
    cellid,
    faceids,
    facedetjac,
    jac,
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
                    facedetjac[faceid],
                    jac,
                    eta,
                )
                massop = interelement_mass_operators(
                    basis,
                    facequads[cellsign, faceid, cellid],
                    facequads[cellsign, nbrfaceids[faceid], nbrcellid],
                    penalty * facedetjac[faceid],
                )
                assemble_face_interelement_condition!(
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

function assemble_face_interelement_condition!(
    sysmatrix,
    nodeids1,
    nodeids2,
    dofspernode,
    tractionop,
    massop,
)
    assemble_cell_matrix!(sysmatrix, nodeids1, dofspernode, +tractionop.nn)
    assemble_cell_matrix!(sysmatrix, nodeids1, dofspernode, +tractionop.nnT)
    assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        nodeids2,
        dofspernode,
        +tractionop.np,
    )
    assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        nodeids2,
        dofspernode,
        -tractionop.pnT,
    )

    assemble_cell_matrix!(sysmatrix, nodeids2, dofspernode, -tractionop.pp)
    assemble_cell_matrix!(sysmatrix, nodeids2, dofspernode, -tractionop.ppT)
    assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids2,
        nodeids1,
        dofspernode,
        -tractionop.pn,
    )
    assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids2,
        nodeids1,
        dofspernode,
        +tractionop.npT,
    )

    assemble_cell_matrix!(sysmatrix, nodeids1, dofspernode, massop.nn)
    assemble_cell_matrix!(sysmatrix, nodeids2, dofspernode, massop.pp)
    assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        nodeids2,
        dofspernode,
        -massop.np,
    )
    assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids2,
        nodeids1,
        dofspernode,
        -massop.pn,
    )
end
