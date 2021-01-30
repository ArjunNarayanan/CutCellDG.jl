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

function displacement_bilinear_form(basis, quad, stiffness, jacobian)
    dim = dimension(basis)
    nf = number_of_basis_functions(basis)
    ndofs = dim * nf
    matrix = zeros(ndofs, ndofs)
    vectosymmconverter = vector_to_symmetric_matrix_converter()
    detjac = prod(jacobian)
    for (p, w) in quad
        grad = transform_gradient(gradient(basis, p), jacobian)
        NK = sum([make_row_matrix(vectosymmconverter[k], grad[:,k]) for k = 1:dim])
        matrix .+= NK' * stiffness * NK * detjac * w
    end
    return matrix
end

function mass_matrix(basis, quad1, quad2, detjac, ndofs)
    numqp = length(quad1)
    @assert length(quad2) == numqp
    nf = number_of_basis_functions(basis)
    totaldofs = ndofs * nf
    matrix = zeros(totaldofs, totaldofs)
    for qpidx = 1:numqp
        p1, w1 = quad1[qpidx]
        p2, w2 = quad2[qpidx]
        @assert w1 ≈ w2

        vals1 = basis(p1)
        vals2 = basis(p2)

        NI1 = interpolation_matrix(vals1, ndofs)
        NI2 = interpolation_matrix(vals2, ndofs)

        matrix .+= NI1' * NI2 * detjac * w1
    end
    return matrix
end

function mass_matrix(basis, quad, detjac, ndofs)
    return mass_matrix(basis,quad,quad,detjac,ndofs)
end

function component_mass_matrix(basis, quad1, quad2, components, scale, ndofs)
    numqp = length(quad1)
    @assert length(quad2) == length(scale) == size(components)[2] == numqp
    nf = number_of_basis_functions(basis)
    dim = dimension(basis)
    totaldofs = ndofs * nf
    matrix = zeros(totaldofs, totaldofs)
    for qpidx = 1:numqp
        p1, w1 = quad1[qpidx]
        p2, w2 = quad2[qpidx]
        @assert w1 ≈ w2

        component = components[:, qpidx]
        projector = component * component'

        vals1 = basis(p1)
        vals2 = basis(p2)

        NI1 = interpolation_matrix(vals1, dim)
        NI2 = make_row_matrix(projector, vals2)

        matrix .+= NI1' * NI2 * scale[qpidx] * w1
    end
    return matrix
end

function uniform_component_mass_matrix(basis, quad, component, scale, ndofs)
    numqp = length(quad)
    extcomponents = repeat(component,inner=(1,numqp))
    extscale = repeat([scale],numqp)
    return component_mass_matrix(basis,quad,quad,extcomponents,extscale,ndofs)
end
