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
