function boundary_traction_component_operator(
    basis,
    quad,
    component,
    normal,
    stiffness,
    dim,
    facedetjac,
    jac,
    vectosymmconverter,
)

    numqp = length(quad)
    normals = repeat(normal, inner = (1, numqp))
    components = repeat(component, inner = (1,numqp))
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
