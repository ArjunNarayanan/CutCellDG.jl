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
    nptop = uniform_surface_traction_operator(
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
    pntop = uniform_surface_traction_operator(
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
    pptop = uniform_surface_traction_operator(
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

function interface_mass_operators(basis, quad1, quad2, dim, facescale)
    nn = mass_matrix(basis, quad1, quad1, dim, facescale)
    np = mass_matrix(basis, quad1, quad2, dim, facescale)
    pn = mass_matrix(basis, quad2, quad1, dim, facescale)
    pp = mass_matrix(basis, quad2, quad2, dim, facescale)

    return InterfaceMassOperatorValues(nn, np, pn, pp)
end
