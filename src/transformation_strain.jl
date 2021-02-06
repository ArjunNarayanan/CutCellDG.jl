function plane_strain_transformation_stress(lambda, mu, theta)
    pt = (lambda + 2mu / 3) * theta
    transfstress = [pt, pt, 0.0]
    return transfstress
end

function bulk_transformation_rhs(
    transfstress,
    basis,
    quad,
    dim,
    jac,
    detjac,
    vectosymmconverter,
)

    nf = number_of_basis_functions(basis)
    ndofs = dim * nf
    rhs = zeros(ndofs)

    for (p, w) in quad
        grad = transform_gradient(gradient(basis, p), jac)
        NK = sum([
            make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim
        ])
        rhs .+= NK' * transfstress * detjac * w
    end
    return rhs
end

function coherent_interface_transformation_linear_form(
    transfstress,
    basis,
    quad,
    normals,
    dim,
    scalearea,
    vectosymmconverter,
)

    numqp = length(quad)
    @assert size(normals) == (dim, numqp)
    @assert length(scalearea) == numqp

    nf = number_of_basis_functions(basis)
    totalndofs = dim * nf
    rhs = zeros(totalndofs)

    for qpidx = 1:numqp
        p, w = quad[qpidx]
        vals = basis(p)
        NI = interpolation_matrix(vals, dim)

        normal = normals[:, qpidx]
        NK = sum([normal[k] * vectosymmconverter[k]' for k = 1:dim])

        rhs .+= NI' * NK * transfstress * scalearea[qpidx] * w
    end
    return rhs
end

function component_interface_transformation_linear_form(
    transfstress,
    basis,
    quad,
    components,
    normals,
    dim,
    scalearea,
    vectosymmconverter,
)

    numqp = length(quad)
    @assert size(components) == (dim, numqp)
    @assert size(normals) == (dim, numqp)
    @assert length(scalearea) == numqp
    nf = number_of_basis_functions(basis)
    ndofs = dim * nf
    rhs = zeros(ndofs)

    for qpidx = 1:numqp
        p, w = quad[qpidx]

        vals = basis(p)
        normal = normals[:, qpidx]
        component = components[:, qpidx]

        projector = component * component'
        NK = sum([normal[k] * vectosymmconverter[k]' for k = 1:dim])
        NI = interpolation_matrix(vals, dim)

        rhs .+= NI' * projector * NK * transfstress * scalearea[qpidx] * w
    end
    return rhs
end
