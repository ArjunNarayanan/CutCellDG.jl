struct LevelSet
    interpolater::Any
    coefficients::Any
    mesh::Any
    dofspernode::Any
end

function LevelSet(distancefunc, mesh, basis::LagrangeTensorProductBasis)
    dofspernode = 1
    interpolater = InterpolatingPolynomial(1, basis)
    coeffs = interpolating_levelset_coefficients(distancefunc, mesh)
    LevelSet(interpolater, coeffs, mesh, dofspernode)
end

function LevelSet(distancefunc, mesh, basis::HermiteTensorProductBasis, quad)
    dim = dimension(mesh)
    dofspernode = hermite_dofs_per_node(dim)
    interpolater = InterpolatingPolynomial(1, basis)
    coeffs = variational_levelset_coefficients(
        distancefunc,
        basis,
        quad,
        mesh,
        dofspernode,
    )
    LevelSet(interpolater, coeffs, mesh, dofspernode)
end

function Base.show(io::IO, levelset::LevelSet)
    p = order(levelset)
    numcoefficients = length(coefficients(levelset))
    dim = dimension(background_mesh(levelset))
    interpolatertype = typeof(basis(interpolater(levelset)))
    str =
        "LevelSet\n\tDimension : $dim\n\tOrder : $p" *
        "\n\tInterpolater Type : $interpolatertype"
    "\n\tTotal Num. Coefficients = $numcoefficients"
    print(io, str)
end

function order(poly::InterpolatingPolynomial)
    return PolynomialBasis.order(poly.basis)
end

function order(levelset::LevelSet)
    return order(interpolater(levelset))
end

function coefficients(levelset::LevelSet)
    return levelset.coefficients
end

function interpolating_levelset_coefficients(distancefunc, mesh)
    nodalcoordinates = nodal_coordinates(mesh)
    return distancefunc(nodalcoordinates)
end

function variational_levelset_coefficients(
    distancefunc,
    basis,
    quad,
    mesh,
    dofspernode,
)
    sysmatrix = SystemMatrix()
    sysrhs = SystemRHS()

    assemble_hermite_mass_matrix!(sysmatrix, basis, quad, mesh, dofspernode)
    assemble_hermite_linear_form!(
        sysrhs,
        distancefunc,
        basis,
        quad,
        mesh,
        dofspernode,
    )

    K = sparse_operator(sysmatrix, mesh, dofspernode)
    R = rhs_vector(sysrhs, mesh, dofspernode)

    coeffs = K \ R
    return coeffs
end

function coefficients(levelset::LevelSet, cellid)
    mesh = background_mesh(levelset)
    dofspernode = dofs_per_node(levelset)
    nodeids = nodal_connectivity(mesh, cellid)
    edofs = element_dofs(nodeids, dofspernode)
    return coefficients(levelset)[edofs]
end

function dofs_per_node(levelset::LevelSet)
    return levelset.dofspernode
end

function background_mesh(levelset::LevelSet)
    return levelset.mesh
end

function interpolater(levelset::LevelSet)
    return levelset.interpolater
end

function load_coefficients!(levelset::LevelSet, cellid)
    update!(interpolater(levelset), coefficients(levelset, cellid))
end

function update_coefficients!(levelset::LevelSet, cellid, coeffs)
    mesh = background_mesh(levelset)
    nodeids = nodal_connectivity(mesh, cellid)
    dofspernode = dofs_per_node(levelset)
    edofs = element_dofs(nodeids, dofspernode)
    levelset.coefficients[edofs] .= coeffs
end

function update_coefficients!(levelset::LevelSet, coeffs)
    levelset.coefficients[:] .= coeffs
end

function (levelset::LevelSet)(x)
    levelset.interpolater(x)
end
