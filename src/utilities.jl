function interpolation_points(basis::TensorProductBasis)
    return basis.points
end

function cell_connectivity(mesh::UniformMesh)
    ncells = number_of_elements(mesh)
    nfaces = faces_per_cell(mesh)
    connectivity = zeros(Int, nfaces, ncells)
    for cellid = 1:ncells
        connectivity[:, cellid] .= neighbors(mesh, cellid)
    end
    return connectivity
end

function dimension(mesh::UniformMesh)
    return CartesianMesh.dimension(mesh)
end

function construct_cell_maps(mesh::UniformMesh)
    ncells = number_of_elements(mesh)
    return [CellMap(element(mesh, i)...) for i = 1:ncells]
end

function reference_corner(mesh::UniformMesh)
    return mesh.x0
end

function widths(mesh::UniformMesh)
    return mesh.widths
end

function cell_sign_to_row(s)
    (s == -1 || s == +1) || error("Use ±1 to index into rows (i.e. phase), got index = $s")
    row = s == +1 ? 1 : 2
    return row
end

function levelset_coefficients(distancefunc,mesh)
    nodalcoordinates = nodal_coordinates(mesh)
    return distancefunc(nodalcoordinates)
end

function levelset_normals(levelset, points, invjac)
    npts = size(points)[2]
    if npts == 0
        return []
    else
        g = hcat([gradient(levelset, points[:, i])' for i = 1:npts]...)
        normals = diagm(invjac) * g
        normalize_normals!(normals)
        return normals
    end
end

function normalize_normals!(normals)
    dim,npts = size(normals)
    for i = 1:npts
        n = normals[:,i]
        normals[:,i] .= n/norm(n)
    end
end

function points(quad::QuadratureRule)
    return quad.points
end

function weights(quad::QuadratureRule)
    return quad.weights
end

function opposite_face(faceid)
    if faceid == 1
        return 3
    elseif faceid == 2
        return 4
    elseif faceid == 3
        return 1
    elseif faceid == 4
        return 2
    else
        error("Expected faceid ∈ {1,2,3,4} got faceid = $faceid")
    end
end

function jacobian(mesh)
    return jacobian(background_mesh(mesh))
end

function inverse_jacobian(mesh)
    return jacobian(background_mesh(mesh))
end

function dimension(mesh)
    return dimension(background_mesh(mesh))
end

function face_determinant_jacobian(mesh)
    return face_determinant_jacobian(background_mesh(mesh))
end

function cell_connectivity(mesh,faceid,cellid)
    return cell_connectivity(background_mesh(mesh),faceid,cellid)
end

function number_of_basis_functions(basis::TensorProductBasis{dim,T,NF}) where {dim,T,NF}

    return NF
end

function reference_face_normals()
    n1 = [0.,-1.]
    n2 = [1.,0.]
    n3 = [0.,1.]
    n4 = [-1.,0.]
    return [n1,n2,n3,n4]
end

function reference_bottom_face_midpoint()
    [0.0, -1.0]
end

function reference_right_face_midpoint()
    [1.0, 0.0]
end

function reference_top_face_midpoint()
    [0.0, 1.0]
end

function reference_left_face_midpoint()
    [-1.0, 0.0]
end

function reference_face_midpoints()
    [
        reference_bottom_face_midpoint(),
        reference_right_face_midpoint(),
        reference_top_face_midpoint(),
        reference_left_face_midpoint(),
    ]
end

function check_cellsign(cellsign)
    @assert cellsign == +1 || cellsign == -1 || cellsign == 0
end
