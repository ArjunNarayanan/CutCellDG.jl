function vector_to_symmetric_matrix_converter()
    E1 = [
        1.0 0.0
        0.0 0.0
        0.0 1.0
    ]
    E2 = [
        0.0 0.0
        0.0 1.0
        1.0 0.0
    ]
    return [E1, E2]
end

function make_row_matrix(matrix, vals)
    return hcat([v * matrix for v in vals]...)
end

function interpolation_matrix(vals, ndofs)
    return make_row_matrix(diagm(ones(ndofs)), vals)
end

function transform_gradient(gradf, jacobian)
    return gradf / Diagonal(jacobian)
end

function displacement_bilinear_form(
    basis,
    quad,
    stiffness,
    jacobian,
    dim,
    vectosymmconverter,
)
    nf = number_of_basis_functions(basis)
    ndofs = dim * nf
    matrix = zeros(ndofs, ndofs)
    detjac = prod(jacobian)
    for (p, w) in quad
        grad = transform_gradient(gradient(basis, p), jacobian)
        NK = sum([
            make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim
        ])
        matrix .+= NK' * stiffness * NK * detjac * w
    end
    return matrix
end

function mass_matrix(basis, quad, ndofs, scale)
    nf = number_of_basis_functions(basis)
    totaldofs = nf * ndofs
    matrix = zeros(totaldofs, totaldofs)
    for (p, w) in quad
        vals = basis(p)

        NI = interpolation_matrix(vals, ndofs)
        matrix .+= NI' * NI * scale * w
    end
    return matrix
end

function mass_matrix(
    basis,
    quad1,
    quad2,
    ndofs,
    facescale;
    weight_tolerance = 1e-3,
)
    numqp = length(quad1)
    @assert length(quad2) == length(facescale) == numqp
    nf = number_of_basis_functions(basis)
    totaldofs = ndofs * nf
    matrix = zeros(totaldofs, totaldofs)
    for qpidx = 1:numqp
        p1, w1 = quad1[qpidx]
        p2, w2 = quad2[qpidx]
        @assert isapprox(w1, w2, atol = weight_tolerance)

        vals1 = basis(p1)
        vals2 = basis(p2)

        NI1 = interpolation_matrix(vals1, ndofs)
        NI2 = interpolation_matrix(vals2, ndofs)

        matrix .+= NI1' * NI2 * facescale[qpidx] * w1
    end
    return matrix
end

function component_mass_matrix(
    basis,
    quad1,
    quad2,
    components,
    ndofs,
    facescale;
    weight_tolerance = 1e-3,
)
    numqp = length(quad1)
    @assert length(quad2) == length(facescale) == numqp
    @assert size(components) == (ndofs, numqp)
    nf = number_of_basis_functions(basis)
    totaldofs = ndofs * nf
    matrix = zeros(totaldofs, totaldofs)
    for qpidx = 1:numqp
        p1, w1 = quad1[qpidx]
        p2, w2 = quad2[qpidx]
        @assert isapprox(w1, w2, atol = weight_tolerance)

        component = components[:, qpidx]
        projector = component * component'

        vals1 = basis(p1)
        vals2 = basis(p2)

        NI1 = interpolation_matrix(vals1, ndofs)
        NI2 = make_row_matrix(projector, vals2)

        matrix .+= NI1' * NI2 * facescale[qpidx] * w1
    end
    return matrix
end

function surface_traction_component_operator(
    basis,
    quad1,
    quad2,
    components,
    normals,
    stiffness,
    dim,
    scalearea,
    jac,
    vectosymmconverter;
    weight_tolerance = 1e-3,
)
    numqp = length(quad1)
    @assert length(quad2) ==
            size(normals)[2] ==
            length(scalearea) ==
            size(components)[2] ==
            numqp
    @assert size(normals)[1] == size(components)[1] == dim

    nf = number_of_basis_functions(basis)
    totalndofs = dim * nf
    matrix = zeros(totalndofs, totalndofs)

    for qpidx = 1:numqp
        p1, w1 = quad1[qpidx]
        p2, w2 = quad2[qpidx]
        @assert isapprox(w1, w2, atol = weight_tolerance)

        vals = basis(p1)
        grad = transform_gradient(gradient(basis, p2), jac)

        normal = normals[:, qpidx]
        component = components[:, qpidx]
        projector = component * component'

        NK = sum([
            make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim
        ])
        N = sum([normal[k] * vectosymmconverter[k]' for k = 1:dim])
        NI = interpolation_matrix(vals, dim)

        matrix .+= NI' * projector * N * stiffness * NK * scalearea[qpidx] * w1
    end
    return matrix
end

function surface_traction_operator(
    basis,
    quad1,
    quad2,
    normals,
    stiffness,
    dim,
    scalearea,
    jac,
    vectosymmconverter;
    weight_tolerance = 1e-3,
)

    numqp = length(quad1)
    @assert length(quad2) == size(normals)[2] == length(scalearea) == numqp
    @assert size(normals)[1] == dim

    nf = number_of_basis_functions(basis)
    totalndofs = dim * nf
    matrix = zeros(totalndofs, totalndofs)

    for qpidx = 1:numqp
        p1, w1 = quad1[qpidx]
        p2, w2 = quad2[qpidx]
        @assert isapprox(w1, w2, atol = weight_tolerance)

        vals = basis(p1)
        grad = transform_gradient(gradient(basis, p2), jac)
        normal = normals[:, qpidx]
        NK = sum([
            make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim
        ])
        N = sum([normal[k] * vectosymmconverter[k]' for k = 1:dim])
        NI = interpolation_matrix(vals, dim)

        matrix .+= NI' * N * stiffness * NK * scalearea[qpidx] * w1
    end
    return matrix
end

function linear_form(rhsfunc, basis, quad, cellmap, ndofs, detjac)
    nf = number_of_basis_functions(basis)
    rhs = zeros(ndofs * nf)
    for (p, w) in quad
        vals = rhsfunc(cellmap(p))
        @assert length(vals) == ndofs
        N = interpolation_matrix(basis(p), ndofs)
        rhs .+= N' * vals * detjac * w
    end
    return rhs
end

function component_linear_form(
    rhsfunc,
    basis,
    quad,
    component,
    cellmap,
    ndofs,
    detjac,
)
    @assert length(component) == ndofs
    nf = number_of_basis_functions(basis)
    rhs = zeros(ndofs * nf)
    for (p, w) in quad
        vals = rhsfunc(cellmap(p))
        N = interpolation_matrix(basis(p), ndofs)
        rhs .+= vals * N' * component * detjac * w
    end
    return rhs
end
