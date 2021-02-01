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

    for cellid = 1:ncells
        cellsign = cell_sign(mesh,cellid)
        check_cellsign(cellsign)
        if cellsign == 0
            nodeids1 = nodal_connectivity(mesh,+1,cellid)
            nodeids2 = nodal_connectivity(mesh,-1,cellid)

            quad1 = interfacequads[+1,cellid]
            quad2 = interfacequads[-1,cellid]

            normals = interface_normals(interfacequads,cellid)

            top = coherent_traction_operators(basis,quad1,quad2,normals,)
        end
    end
end
