struct InterfaceTractionOperatorValues
    nn::Any
    np::Any
    pn::Any
    pp::Any
    nnT::Any
    npT::Any
    pnT::Any
    ppT::Any
    function InterfaceTractionOperatorValues(nntop, nptop, pntop, pptop, eta)
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

function coherent_traction_operators(
    basis,
    quad1,
    quad2,
    normals,
    stiffness1,
    stiffness2,
    dim,
    scalearea,
    jac,
    vectosymmconverter,
    eta,
)
    nntop = surface_traction_operator(
        basis,
        quad1,
        quad1,
        normals,
        stiffness1,
        dim,
        scalearea,
        jac,
        vectosymmconverter,
    )
    nptop = surface_traction_operator(
        basis,
        quad1,
        quad2,
        normals,
        stiffness2,
        dim,
        scalearea,
        jac,
        vectosymmconverter,
    )
    pntop = surface_traction_operator(
        basis,
        quad2,
        quad1,
        normals,
        stiffness1,
        dim,
        scalearea,
        jac,
        vectosymmconverter,
    )
    pptop = surface_traction_operator(
        basis,
        quad2,
        quad2,
        normals,
        stiffness2,
        dim,
        scalearea,
        jac,
        vectosymmconverter,
    )

    return InterfaceTractionOperatorValues(nntop, nptop, pntop, pptop, eta)
end

function component_traction_operators(
    basis,
    quad1,
    quad2,
    components,
    normals,
    stiffness1,
    stiffness2,
    dim,
    scalearea,
    jac,
    vectosymmconverter,
    eta,
)

    nntop = surface_traction_component_operator(
        basis,
        quad1,
        quad1,
        components,
        normals,
        stiffness1,
        dim,
        scalearea,
        jac,
        vectosymmconverter,
    )
    nptop = surface_traction_component_operator(
        basis,
        quad1,
        quad2,
        components,
        normals,
        stiffness2,
        dim,
        scalearea,
        jac,
        vectosymmconverter,
    )
    pntop = surface_traction_component_operator(
        basis,
        quad2,
        quad1,
        components,
        normals,
        stiffness1,
        dim,
        scalearea,
        jac,
        vectosymmconverter,
    )
    pptop = surface_traction_component_operator(
        basis,
        quad2,
        quad2,
        components,
        normals,
        stiffness2,
        dim,
        scalearea,
        jac,
        vectosymmconverter,
    )

    return InterfaceTractionOperatorValues(nntop, nptop, pntop, pptop, eta)
end

struct InterfaceMassOperatorValues
    nn::Any
    np::Any
    pn::Any
    pp::Any
    function InterfaceMassOperatorValues(nnmop, npmop, pnmop, ppmop)
        nn = vec(nnmop)
        np = vec(npmop)
        pn = vec(pnmop)
        pp = vec(ppmop)

        new(nn, np, pn, pp)
    end
end

function coherent_mass_operators(basis, quad1, quad2, dim, facescale)
    nn = mass_matrix(basis, quad1, quad1, dim, facescale)
    np = mass_matrix(basis, quad1, quad2, dim, facescale)
    pn = mass_matrix(basis, quad2, quad1, dim, facescale)
    pp = mass_matrix(basis, quad2, quad2, dim, facescale)

    return InterfaceMassOperatorValues(nn, np, pn, pp)
end

function component_mass_operators(
    basis,
    quad1,
    quad2,
    components,
    dim,
    facescale,
)
    nn = component_mass_matrix(basis, quad1, quad1, components, dim, facescale)
    np = component_mass_matrix(basis, quad1, quad2, components, dim, facescale)
    pn = component_mass_matrix(basis, quad2, quad1, components, dim, facescale)
    pp = component_mass_matrix(basis, quad2, quad2, components, dim, facescale)

    return InterfaceMassOperatorValues(nn, np, pn, pp)
end

function assemble_interface_condition!(
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
