struct MergeMapper
    cellmaps::Any
    function MergeMapper()
        south = cell_map_to_south()
        east = cell_map_to_east()
        north = cell_map_to_north()
        west = cell_map_to_west()
        southeast = cell_map_to_south_east()
        northeast = cell_map_to_north_east()
        northwest = cell_map_to_north_west()
        southwest = cell_map_to_south_west()
        new([
            south,
            east,
            north,
            west,
            southeast,
            northeast,
            northwest,
            southwest,
        ])
    end
end

function Base.getindex(m::MergeMapper, i)
    return m.cellmaps[i]
end

function cell_map_to_south()
    cellmap = CellMap([-1.0, -3.0], [1.0, -1.0])
    return cellmap
end

function cell_map_to_east()
    cellmap = CellMap([1.0, -1.0], [3.0, 1.0])
    return cellmap
end

function cell_map_to_north()
    cellmap = CellMap([-1.0, 1.0], [1.0, 3.0])
    return cellmap
end

function cell_map_to_west()
    cellmap = CellMap([-3.0, -1.0], [-1.0, 1.0])
    return cellmap
end

function cell_map_to_south_east()
    cellmap = CellMap([1.0, -3.0], [3.0, -1.0])
    return cellmap
end

function cell_map_to_north_east()
    cellmap = CellMap([1.0, 1.0], [3.0, 3.0])
    return cellmap
end

function cell_map_to_north_west()
    cellmap = CellMap([-3.0, 1.0], [-1.0, 3.0])
    return cellmap
end

function cell_map_to_south_west()
    cellmap = CellMap([-3.0, -3.0], [-1.0, -1.0])
    return cellmap
end

function map_quadrature(quad, mergemapper, mapid)
    mappedpoints = mergemapper[mapid](points(quad))
    return QuadratureRule(mappedpoints, weights(quad))
end

function map_and_update_cell_quadrature!(
    cellquads,
    s,
    cellid,
    mergemapper,
    mapid,
)
    quad = cellquads[s, cellid]
    newquad = map_quadrature(quad, mergemapper, mapid)
    update_cell_quadrature!(cellquads, s, cellid, newquad)
end

function map_and_update_interface_quadrature!(
    interfacequads,
    s,
    cellid,
    mergemapper,
    mapid,
)
    quad = interfacequads[s, cellid]
    newquad = map_quadrature(quad, mergemapper, mapid)
    update_interface_quadrature!(interfacequads, s, cellid, newquad)
end

function map_and_update_face_quadrature!(
    facequads,
    s,
    cellid,
    mergemapper,
    mapid,
)
    nfaces = number_of_faces_per_cell(facequads)
    for faceid = 1:nfaces
        quad = facequads[s, faceid, cellid]
        newquad = map_quadrature(quad, mergemapper, mapid)
        update_face_quadrature!(facequads, s, faceid, cellid, newquad)
    end
end

function merged_with_cell(mergedwithcell, s, cellid)
    row = cell_sign_to_row(s)
    return mergedwithcell[row, cellid]
end

function merge_cells!(mergedwithcell, s, mergeto, mergefrom)
    row = cell_sign_to_row(s)
    ncells = size(mergedwithcell)[2]
    @assert 1 <= mergeto <= ncells
    @assert 1 <= mergefrom <= ncells

    mergedwithcell[row, mergefrom] = mergeto
end

function update_quadrature_area!(areas, cellquads, s, cellid)
    row = cell_sign_to_row(s)
    areas[row, cellid] = sum(weights(cellquads[s, cellid]))
end

function quadrature_areas(cellquads, cutmesh)
    ncells = number_of_cells(cutmesh)
    areas = zeros(2, ncells)
    for cellid = 1:ncells
        s = cell_sign(cutmesh, cellid)
        if s == +1 || s == 0
            update_quadrature_area!(areas, cellquads, +1, cellid)
        end
        if s == -1 || s == 0
            update_quadrature_area!(areas, cellquads, -1, cellid)
        end
    end
    return areas
end

function mark_tiny_cell!(istiny, quadareas, s, cellid, tinyarea)
    row = cell_sign_to_row(s)
    if quadareas[row, cellid] <= tinyarea
        istiny[row, cellid] = true
    end
end

function is_tiny_cell(cutmesh, quadareas, tinyarea)
    ncells = number_of_cells(cutmesh)
    istiny = zeros(Bool, 2, ncells)

    for cellid = 1:ncells
        s = cell_sign(cutmesh, cellid)

        if s == +1 || s == 0
            mark_tiny_cell!(istiny, quadareas, +1, cellid, tinyarea)
        end
        if s == -1 || s == 0
            mark_tiny_cell!(istiny, quadareas, -1, cellid, tinyarea)
        end
    end

    return istiny
end

function merge_cell_with_suitable_neighbor!(
    mergedwithcell,
    cellquads,
    facequads,
    interfacequads,
    cutmesh,
    istinycell,
    cellsign,
    cellid,
    nfaces,
    mergemapper,
)
    row = cell_sign_to_row(cellsign)

    nbrcellids = cell_connectivity(background_mesh(cutmesh), :, cellid)
    nbrcellsign =
        [nbrid == 0 ? 2 : cell_sign(cutmesh, nbrid) for nbrid in nbrcellids]
    nbrislarge =
        [nbrid == 0 ? false : !istinycell[row, nbrid] for nbrid in nbrcellids]
    nbr_has_same_phase = [ns == 0 || ns == cellsign for ns in nbrcellsign]

    faceid = findfirst(nbr_has_same_phase .& nbrislarge)

    !isnothing(faceid) || error(
        "Could not find appropriate merge direction for cell $cellid, cellsign $cellsign",
    )
    oppositeface = opposite_face(faceid)
    mergecellid = nbrcellids[faceid]

    merge_cells!(mergedwithcell, cellsign, mergecellid, cellid)
    map_and_update_cell_quadrature!(
        cellquads,
        cellsign,
        cellid,
        mergemapper,
        oppositeface,
    )
    map_and_update_interface_quadrature!(
        interfacequads,
        cellsign,
        cellid,
        mergemapper,
        oppositeface,
    )
    map_and_update_face_quadrature!(
        facequads,
        cellsign,
        cellid,
        mergemapper,
        oppositeface,
    )
end

struct MergedMesh
    mergedwithcell::Any
    cutmesh::Any
    nodelabeltonodeid::Any
    numnodes::Any
    function MergedMesh(cutmesh, mergedwithcell)
        activenodelabels = active_node_labels(cutmesh, mergedwithcell)
        maxlabel = maximum(activenodelabels)
        nodelabeltonodeid = zeros(Int, maxlabel)
        for (idx, label) in enumerate(activenodelabels)
            nodelabeltonodeid[label] = idx
        end
        numnodes = length(activenodelabels)
        new(mergedwithcell, cutmesh, nodelabeltonodeid, numnodes)
    end
end

function merge_tiny_cells_in_mesh!(
    cutmesh,
    cellquads,
    facequads,
    interfacequads;
    tinyratio = 0.2,
)
    ncells = number_of_cells(cutmesh)

    mergemapper = MergeMapper()
    mergedwithcell = vcat((1:ncells)', (1:ncells)')
    quadareas = quadrature_areas(cellquads, cutmesh)
    tinyarea = tinyratio * sum(weights(uniform_cell_quadrature(cellquads)))
    istinycell = is_tiny_cell(cutmesh, quadareas, tinyarea)
    nfaces = number_of_faces_per_cell(facequads)

    for cellid = 1:ncells
        s = cell_sign(cutmesh, cellid)
        if s == +1 || s == 0
            row = cell_sign_to_row(+1)
            if istinycell[row, cellid]
                merge_cell_with_suitable_neighbor!(
                    mergedwithcell,
                    cellquads,
                    facequads,
                    interfacequads,
                    cutmesh,
                    istinycell,
                    +1,
                    cellid,
                    nfaces,
                    mergemapper,
                )
            end
        end
        if s == -1 || s == 0
            row = cell_sign_to_row(-1)
            if istinycell[row, cellid]
                merge_cell_with_suitable_neighbor!(
                    mergedwithcell,
                    cellquads,
                    facequads,
                    interfacequads,
                    cutmesh,
                    istinycell,
                    -1,
                    cellid,
                    nfaces,
                    mergemapper,
                )
            end
        end
    end
    return mergedwithcell
end

function background_mesh(mergedmesh::MergedMesh)
    return mergedmesh.cutmesh
end

function number_of_nodes(mergedmesh::MergedMesh)
    return mergedmesh.numnodes
end

function number_of_cells(mergedmesh::MergedMesh)
    return number_of_cells(mergedmesh.cutmesh)
end

function cell_map(mergedmesh::MergedMesh,cellsign,cellid)
    row = cell_sign_to_row(cellsign)
    mergecellid = merged_with_cell(mergedmesh.mergedwithcell,cellsign,cellid)
    return cell_map(mergedmesh.cutmesh,mergecellid)
end

function cell_sign(mergedmesh::MergedMesh,cellid)
    return cell_sign(background_mesh(mergedmesh),cellid)
end

function nodal_connectivity(mergedmesh::MergedMesh,cellsign,cellid)
    row = cell_sign_to_row(cellsign)
    mergecellid = merged_with_cell(mergedmesh.mergedwithcell,cellsign,cellid)
    nodelabels = nodal_connectivity(mergedmesh.cutmesh,cellsign,mergecellid)
    return mergedmesh.nodelabeltonodeid[nodelabels]
end

function Base.show(io::IO, mergedmesh::MergedMesh)
    ncells = number_of_cells(mergedmesh)
    numnodes = number_of_nodes(mergedmesh)
    str = "MergedMesh\n\tNum. Cells: $ncells\n\tNum. Nodes: $numnodes"
    print(io, str)
end

function active_node_labels(cutmesh, mergedwithcell)
    numcells = number_of_cells(cutmesh)

    activenodeids = Int[]

    for cellid = 1:numcells
        s = cell_sign(cutmesh, cellid)
        if s == +1 || s == 0
            mergecellid = merged_with_cell(mergedwithcell, +1, cellid)
            append!(activenodeids, nodal_connectivity(cutmesh, +1, mergecellid))
        end
        if s == -1 || s == 0
            mergecellid = merged_with_cell(mergedwithcell, -1, cellid)
            append!(activenodeids, nodal_connectivity(cutmesh, -1, mergecellid))
        end
    end

    sort!(activenodeids)
    unique!(activenodeids)
    return activenodeids
end
