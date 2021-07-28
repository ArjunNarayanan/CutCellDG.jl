struct CGMesh
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
    nfmside::Any
    mesh::Any
    elementsize::Any
end

function CGMesh(mesh, nodesperelement)
    dim = dimension(mesh)
    cellmaps = construct_cell_maps(mesh)
    x0 = reference_corner(mesh)
    meshwidths = mesh_widths(mesh)
    nelements = elements_per_mesh_side(mesh)
    elementsize = meshwidths ./ nelements
    nfeside = nodes_per_element_side(nodesperelement, dim)
    nfmside = nodes_per_mesh_side(nelements, nfeside)
    numnodes = prod(nfmside)

    nodalcoordinates =
        cg_nodal_coordinates(x0, mesh_widths(mesh), nelements, nfmside)
    nodalconnectivity =
        cg_nodal_connectivity(nfmside, nfeside, nodesperelement, nelements)
    cellconnectivity = cell_connectivity(mesh)
    numcells = number_of_elements(mesh)
    CGMesh(
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
        nfmside,
        mesh,
        elementsize,
    )
end

function CGMesh(x0, meshwidths, nelements, nodesperelement)
    mesh = UniformMesh(x0, meshwidths, nelements)
    return CGMesh(mesh, nodesperelement)
end

function Base.show(io::IO, mesh::CGMesh)
    dim = dimension(mesh)
    x0 = reference_corner(mesh)
    nelements = elements_per_mesh_side(mesh)
    meshwidths = mesh_widths(mesh)
    ncells = number_of_cells(mesh)
    nodesperelement = nodes_per_element(mesh)
    nnodes = number_of_nodes(mesh)
    nf = nodes_per_element(mesh)

    str =
        "CGMesh\n\tDimension : $dim\n\tCorner : $x0\n\tWidth : $meshwidths\n\t" *
        "Elements/Side : $nelements\n\tNum. Cells : $ncells\n\t" *
        "Nodes/Element : $nodesperelement\n\t" *
        "Num. Nodes : $nnodes"
    print(io, str)
end

function element_size(mesh::CGMesh)
    return mesh.elementsize
end

function dimension(mesh::CGMesh)
    return mesh.dim
end

function reference_corner(mesh::CGMesh)
    return mesh.x0
end

function mesh_widths(mesh::CGMesh)
    return mesh.meshwidths
end

function elements_per_mesh_side(mesh::CGMesh)
    return mesh.nelements
end

function number_of_cells(mesh::CGMesh)
    mesh.numcells
end

function number_of_nodes(mesh::CGMesh)
    return mesh.numnodes
end

function nodes_per_element(mesh::CGMesh)
    return mesh.nodesperelement
end

function nodal_connectivity(mesh::CGMesh)
    return mesh.nodalconnectivity
end

function nodal_connectivity(mesh::CGMesh, cellid::Int)
    return mesh.nodalconnectivity[:, cellid]
end

function nodes_per_mesh_side(mesh::CGMesh)
    return mesh.nfmside
end

function nodal_coordinates(mesh::CGMesh)
    return mesh.nodalcoordinates
end

function background_mesh(mesh::CGMesh)
    return mesh.mesh
end

function cell_map(mesh::CGMesh, cellid)
    return mesh.cellmaps[cellid]
end

function jacobian(mesh::CGMesh)
    return jacobian(cell_map(mesh, 1))
end

function inverse_jacobian(mesh::CGMesh)
    return inverse_jacobian(cell_map(mesh, 1))
end
################################################################################

function nodes_per_element_side(nodesperelement, dim)
    nfeside = (nodesperelement)^(1.0 / dim)
    @assert isinteger(nfeside)
    return round(Int, nfeside)
end

function nodes_per_mesh_side(nelements, nfeside)
    return (nfeside - 1) * nelements .+ 1
end

function cg_nodal_coordinates(x0, widths, nelements, nfmside)
    dim = length(x0)
    @assert length(x0) == length(widths) == length(nelements) == dim

    if dim == 1
        xrange = range(x0[1], stop = x0[1] + widths[1], length = nfmside[1])
        return Array(transpose(xrange))
    elseif dim == 2
        xrange = range(x0[1], stop = x0[1] + widths[1], length = nfmside[1])
        yrange = range(x0[2], stop = x0[2] + widths[2], length = nfmside[2])

        ycoords = repeat(yrange, outer = nfmside[1])
        xcoords = repeat(xrange, inner = nfmside[2])
        return vcat(xcoords', ycoords')
    else
        error("3D currently not supported")
    end
end

function cg_nodal_connectivity_2d(nfmside, nfeside, nodesperelement, nelements)
    numnodes = prod(nfmside)
    ncells = prod(nelements)

    nodeids = reshape(1:numnodes, reverse(nfmside)...)
    connectivity = zeros(Int, nodesperelement, ncells)
    colstart = 1
    colend = nfeside

    cellid = 1
    for col = 1:nelements[1]
        rowstart = 1
        rowend = nfeside
        for row = 1:nelements[2]
            connectivity[:, cellid] .=
                vec(nodeids[rowstart:rowend, colstart:colend])
            cellid += 1
            rowstart = rowend
            rowend += nfeside - 1
        end
        colstart = colend
        colend += nfeside - 1
    end
    return connectivity
end

function cg_nodal_connectivity_1d(nfmside, nodesperelement, nelements)
    numnodes = prod(nfmside)
    ncells = prod(nelements)

    connectivity = zeros(Int, nodesperelement, ncells)
    nodeids = 1:numnodes
    start = 1
    stop = nodesperelement

    for cellid = 1:ncells
        connectivity[:, cellid] = nodeids[start:stop]
        start = stop
        stop = start + nodesperelement - 1
    end
    return connectivity
end

function cg_nodal_connectivity(nfmside, nfeside, nodesperelement, nelements)
    dim = length(nfmside)
    @assert length(nfmside) == length(nelements)

    if dim == 1
        return cg_nodal_connectivity_1d(nfmside,nodesperelement,nelements)
    elseif dim == 2
        return cg_nodal_connectivity_2d(nfmside,nfeside,nodesperelement,nelements)
    else
        error("3D not supported")
    end
end
