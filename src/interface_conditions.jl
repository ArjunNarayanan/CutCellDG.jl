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

function interelement_mass_operators(basis,quad1,quad2,dim,scale)
    numqp = length(quad1)
    facescale = repeat([scale],numqp)
    return interface_mass_operators(basis,quad1,quad2,dim,facescale)
end
