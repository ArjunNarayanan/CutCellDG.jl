struct CutMesh
    mesh
    cellsign
    cutmeshnodeids
    ncells
    numnodes
    function CutMesh(mesh, cellsign::Vector{Int}, cutmeshnodeids::Matrix{Int})
        ncells = number_of_cells(mesh)
        nummeshnodes = number_of_nodes(mesh)
        @assert length(cellsign) == ncells
        @assert size(cutmeshnodeids) == (2, nummeshnodes)

        numnodes = maximum(cutmeshnodeids)
        new(mesh, cellsign, cutmeshnodeids, ncells, numnodes)
    end
end

function CutMesh(mesh, levelset, levelsetcoeffs; tol = 1e-4, perturbation = 1e-3)
    cellsign = cell_sign!(mesh, levelset, levelsetcoeffs, tol, perturbation)

    posactivenodeids = active_node_ids(mesh, +1, cellsign)
    negactivenodeids = active_node_ids(mesh, -1, cellsign)

    totalnumnodes = number_of_nodes(mesh)
    cutmeshnodeids = cut_mesh_nodeids(posactivenodeids, negactivenodeids, totalnumnodes)
    return CutMesh(mesh, cellsign, cutmeshnodeids)
end

function cell_sign(cutmesh::CutMesh, cellid)
    return cutmesh.cellsign[cellid]
end

function dimension(cutmesh::CutMesh)
    return dimension(background_mesh(cutmesh))
end

function number_of_nodes(cutmesh::CutMesh)
    return cutmesh.numnodes
end

function reference_corner(cutmesh::CutMesh)
    return reference_corner(background_mesh(cutmesh))
end

function widths(cutmesh::CutMesh)
    return widths(background_mesh(cutmesh))
end

function number_of_cells(cutmesh::CutMesh)
    return cutmesh.ncells
end

function background_mesh(cutmesh::CutMesh)
    return cutmesh.mesh
end

function cell_map(cutmesh::CutMesh,cellid)
    return cell_map(background_mesh(cutmesh),cellid)
end

function cell_map(cutmesh::CutMesh,cellsign,cellid)
    return cell_map(cutmesh,cellid)
end

function solution_cell_id(cutmesh::CutMesh,s,cellid)
    return cellid
end

function nodal_connectivity(cutmesh::CutMesh, s, cellid)
    row = cell_sign_to_row(s)
    ncells = cutmesh.ncells
    @assert 1 <= cellid <= ncells
    cs = cell_sign(cutmesh, cellid)
    @assert cs == s || cs == 0 || error("Requested nodeids for sign $s not consistent with cell sign $cs for cellid $cellid")

    ids = nodal_connectivity(background_mesh(cutmesh),cellid)
    nodeids = cutmesh.cutmeshnodeids[row, ids]
    return nodeids
end

function nodal_coordinates(cutmesh::CutMesh)
    return nodal_coordinates(background_mesh(cutmesh))
end

function cell_sign!(mesh, levelset, levelsetcoeffs, tol, perturbation)
    ncells = number_of_cells(mesh)
    cellsign = zeros(Int, ncells)
    xL,xR = [-1.,-1.],[1.,1.]
    for cellid = 1:ncells
        nodeids = nodal_connectivity(mesh,cellid)
        update!(levelset, levelsetcoeffs[nodeids])

        s = sign(levelset, xL, xR, tol=tol)
        if (s == +1 || s == 0 || s == -1)
            cellsign[cellid] = s
        else
            @warn "Perturbing levelset function by perturbation = $perturbation"
            levelsetcoeffs[nodeids] .+= perturbation
            update!(levelset,levelsetcoeffs[nodeids])

            s = sign(levelset, xL, xR, tol=tol)
            if (s == +1 || s == 0 || s == -1)
                cellsign[cellid] = s
            else
                error("Could not determine cell sign after perturbation = $perturbation")
            end
        end
    end
    return cellsign
end

function active_node_ids(mesh,s,cellsign)
    @assert (s == -1 || s == +1)
    activenodeids = Int[]
    numcells = number_of_cells(mesh)
    @assert length(cellsign) == numcells

    for cellid = 1:numcells
        if cellsign[cellid] == s || cellsign[cellid] == 0
            append!(activenodeids, nodal_connectivity(mesh,cellid))
        end
    end
    return activenodeids
end

function cut_mesh_nodeids(posactivenodeids, negactivenodeids, totalnumnodes)
    cutmeshnodeids = zeros(Int, 2, totalnumnodes)
    counter = 1
    for nodeid in posactivenodeids
        cutmeshnodeids[1, nodeid] = counter
        counter += 1
    end
    for nodeid in negactivenodeids
        cutmeshnodeids[2, nodeid] = counter
        counter += 1
    end
    return cutmeshnodeids
end
