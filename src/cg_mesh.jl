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
    mesh
end

function CGMesh(mesh, nodesperelement)
    dim = dimension(mesh)
    cellmaps = construct_cell_maps(mesh)
    x0 = reference_corner(mesh)
    meshwidths = widths(mesh)
    nelements = elements_per_mesh_side(mesh)
    elementsize = meshwidths ./ nelements
    nfeside = nodes_per_element_side(nodesperelement)
    nfmside = nodes_per_mesh_side(nelements, nfeside)
    numnodes = prod(nfmside)

    nodalcoordinates =
        cg_nodal_coordinates(x0, widths(mesh), nelements, nfmside)
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
        mesh
    )
end

function CGMesh(x0,meshwidths,nelements,nodesperelement::Int)
    mesh = UniformMesh(x0,meshwidths,nelements)
    return CGMesh(mesh,nodesperelement)
end

function CGMesh(x0,meshwidths,nelements,basis)
    nodesperelement = number_of_basis_functions(basis)
    mesh = UniformMesh(x0,meshwidths,nelements)
    return CGMesh(mesh,nodesperelement)
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

function cell_map(mesh::CGMesh,cellid)
    return mesh.cellmaps[cellid]
end

################################################################################

function nodes_per_element_side(nodesperelement)
    nfeside = sqrt(nodesperelement)
    @assert isinteger(nfeside)
    return round(Int, nfeside)
end

function nodes_per_mesh_side(nelements, nfeside)
    return (nfeside - 1) * nelements .+ 1
end

function cg_nodal_coordinates(x0, widths, nelements, nfmside)
    @assert length(x0) == length(widths) == length(nelements) == 2

    xrange = range(x0[1], stop = x0[1] + widths[1], length = nfmside[1])
    yrange = range(x0[2], stop = x0[2] + widths[2], length = nfmside[2])

    ycoords = repeat(yrange, outer = nfmside[1])
    xcoords = repeat(xrange, inner = nfmside[2])
    return vcat(xcoords', ycoords')
end

function cg_nodal_connectivity(nfmside, nfeside, nodesperelement, nelements)
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
