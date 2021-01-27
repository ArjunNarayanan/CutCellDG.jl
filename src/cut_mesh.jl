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
        nelmts = count(activecells)
        new(mesh, cellsign, activecells, cutmeshnodeids, ncells, numnodes)
    end
end

function CutMesh(levelset::InterpolatingPolynomial, levelsetcoeffs, mesh; tol = 1e-4, perturbation = 1e-3)
    nodalconnectivity = nodal_connectivity(mesh)
    cellsign = cell_sign!(levelset, levelsetcoeffs, nodalconnectivity, tol, perturbation)

    posactivenodeids = active_node_ids(+1, cellsign, nodalconnectivity)
    negactivenodeids = active_node_ids(-1, cellsign, nodalconnectivity)

    totalnumnodes = number_of_nodes(mesh)
    cutmeshnodeids = cut_mesh_nodeids(posactivenodeids, negactivenodeids, totalnumnodes)
    return CutMesh(mesh, cellsign, cutmeshnodeids)
end
