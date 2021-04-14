struct DGMesh
    dim::Any
    cellmaps::Any
    nodalcoordinates::Any
    nodalconnectivity::Any
    cellconnectivity::Any
    numcells::Any
    numnodes::Any
    nodesperelement::Any
    x0::Any
    meshwidths::Any
    nelements::Any
    mesh
end

function DGMesh(mesh, refcoords)
    dim, nodesperelement = size(refcoords)
    @assert dimension(mesh) == dim

    cellmaps = construct_cell_maps(mesh)
    numcells = length(cellmaps)
    numnodes = numcells * nodesperelement
    nodalcoordinates = dg_nodal_coordinates(cellmaps, refcoords)
    nodalconnectivity = dg_nodal_connectivity(numcells, nodesperelement)
    cellconnectivity = cell_connectivity(mesh)
    x0 = reference_corner(mesh)
    meshwidths = widths(mesh)
    nelements = elements_per_mesh_side(mesh)

    DGMesh(
        dim,
        cellmaps,
        nodalcoordinates,
        nodalconnectivity,
        cellconnectivity,
        numcells,
        numnodes,
        nodesperelement,
        x0,
        meshwidths,
        nelements,
        mesh
    )
end

function DGMesh(x0, widths, nelements, basis)
    refpoints = interpolation_points(basis)
    mesh = UniformMesh(x0, widths, nelements)
    return DGMesh(mesh, refpoints)
end

function Base.show(io::IO, dgmesh::DGMesh)
    dim = dimension(dgmesh)
    x0 = reference_corner(dgmesh)
    nelements = elements_per_mesh_side(dgmesh)
    meshwidths = mesh_widths(dgmesh)
    ncells = number_of_cells(dgmesh)
    nodesperelement = nodes_per_element(dgmesh)
    nnodes = number_of_nodes(dgmesh)
    str =
        "DGMesh\n\tDimension : $dim\n\tCorner : $x0\n\tWidth : $meshwidths\n\t" *
        "Elements/Side : $nelements\n\tNum. Cells : $ncells\n\t" *
        "Nodes/Element : $nodesperelement\n\t" *
        "Num. Nodes : $nnodes"
    print(io, str)
end

function elements_per_mesh_side(mesh::DGMesh)
    return mesh.nelements
end

function dimension(dgmesh::DGMesh)
    return dgmesh.dim
end

function number_of_cells(mesh::DGMesh)
    return mesh.numcells
end

function number_of_nodes(mesh::DGMesh)
    return mesh.numnodes
end

function nodes_per_element(mesh::DGMesh)
    return mesh.nodesperelement
end

function reference_corner(mesh::DGMesh)
    return mesh.x0
end

function mesh_widths(mesh::DGMesh)
    return mesh.meshwidths
end

function elements_per_mesh_side(mesh::DGMesh)
    mesh.nelements
end

function background_mesh(mesh::DGMesh)
    return mesh.mesh
end




##########################################################################################

function cell_map(mesh::DGMesh, cellid::Int)
    return mesh.cellmaps[cellid]
end

function nodal_coordinates(mesh::DGMesh)
    return mesh.nodalcoordinates
end

function nodal_connectivity(mesh::DGMesh, cellid::Int)
    return mesh.nodalconnectivity[:, cellid]
end

function nodal_coordinates(mesh::DGMesh, cellid::Int)
    nodeids = nodal_connectivity(mesh, cellid)
    return mesh.nodal_coordinates[:, nodeids]
end

function cell_connectivity(mesh::DGMesh, faceid, cellid)
    return mesh.cellconnectivity[faceid, cellid]
end

function dg_nodal_coordinates(cellmaps, refcoords)
    dim, nodesperelement = size(refcoords)
    numcells = length(cellmaps)
    numnodes = nodesperelement * numcells

    nodalcoordinates = zeros(dim, numnodes)

    for cellid = 1:numcells
        start = (cellid - 1) * nodesperelement + 1
        stop = start + nodesperelement - 1
        cellmap = cellmaps[cellid]

        coords = cellmap(refcoords)

        nodalcoordinates[:, start:stop] .= coords
    end
    return nodalcoordinates
end

function dg_nodal_connectivity(numelements, nodesperelement)
    numnodes = nodesperelement * numelements
    nodeids = 1:numnodes
    nodalconnectivity = reshape(nodeids, nodesperelement, numelements)
    return nodalconnectivity
end

function jacobian(mesh::DGMesh)
    return jacobian(cell_map(mesh, 1))
end

function inverse_jacobian(mesh::DGMesh)
    return inverse_jacobian(cell_map(mesh, 1))
end

function face_determinant_jacobian(mesh::DGMesh)
    return face_determinant_jacobian(cell_map(mesh, 1))
end
