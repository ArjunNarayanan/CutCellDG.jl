function assemble_coherent_interface_condition!(
    sysmatrix,
    basis,
    interfacequads,
    stiffness,
    mesh,
    penalty,
    eta,
)

    check_eta(eta)
    dim = dimension(mesh)
    ncells = number_of_cells(mesh)
    jac = jacobian(mesh)
    vectosymmconverter = vector_to_symmetric_matrix_converter()

    for cellid = 1:ncells
        cellsign = cell_sign(mesh, cellid)
        check_cellsign(cellsign)
        if cellsign == 0
            quad1 = interfacequads[-1, cellid]
            quad2 = interfacequads[+1, cellid]

            normals = interface_normals(interfacequads, cellid)
            scaleareas = interface_scale_areas(interfacequads, cellid)

            top = coherent_traction_operators(
                basis,
                quad1,
                quad2,
                normals,
                stiffness[-1],
                stiffness[+1],
                dim,
                scaleareas,
                jac,
                vectosymmconverter,
                eta,
            )
            mop = coherent_mass_operators(
                basis,
                quad1,
                quad2,
                dim,
                penalty * scaleareas,
            )

            nodeids1 = nodal_connectivity(mesh, -1, cellid)
            nodeids2 = nodal_connectivity(mesh, +1, cellid)

            assemble_interface_condition!(
                sysmatrix,
                nodeids1,
                nodeids2,
                dim,
                top,
                mop,
            )
        end
    end
end

function assemble_incoherent_interface_condition!(
    sysmatrix,
    basis,
    interfacequads,
    stiffness,
    mesh,
    penalty,
    eta,
)

    check_eta(eta)
    dim = dimension(mesh)
    ncells = number_of_cells(mesh)
    jac = jacobian(mesh)
    vectosymmconverter = vector_to_symmetric_matrix_converter()

    for cellid = 1:ncells
        cellsign = cell_sign(mesh, cellid)
        check_cellsign(cellsign)
        if cellsign == 0
            quad1 = interfacequads[-1, cellid]
            quad2 = interfacequads[+1, cellid]

            normals = interface_normals(interfacequads, cellid)
            components = normals
            scaleareas = interface_scale_areas(interfacequads, cellid)

            top = component_traction_operators(
                basis,
                quad1,
                quad2,
                components,
                normals,
                stiffness[-1],
                stiffness[+1],
                dim,
                scaleareas,
                jac,
                vectosymmconverter,
                eta,
            )
            mop = component_mass_operators(
                basis,
                quad1,
                quad2,
                components,
                dim,
                penalty * scaleareas,
            )

            nodeids1 = nodal_connectivity(mesh, -1, cellid)
            nodeids2 = nodal_connectivity(mesh, +1, cellid)

            assemble_interface_condition!(
                sysmatrix,
                nodeids1,
                nodeids2,
                dim,
                top,
                mop,
            )
        end
    end
end
