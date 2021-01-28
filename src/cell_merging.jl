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
        new([south, east, north, west, southeast, northeast, northwest, southwest])
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
    mappedpoints = mergemapper[mapid](quad.points)
    return QuadratureRule(mappedpoints, quad.weights)
end

function map_and_update_cell_quadrature!(cellquads, s, cellid, mergemapper, mapid)
    quad = cellquads[s, cellid]
    newquad = map_quadrature(quad, mergemapper, mapid)
    update_cell_quadrature!(cellquads, s, cellid, newquad)
end

function map_and_update_interface_quadrature!(interfacequads, s, cellid, mergemapper, mapid)
    quad = interfacequads[s, cellid]
    newquad = map_quadrature(quad, mergemapper, mapid)
    update_interface_quadrature!(interfacequads, s, cellid, newquad)
end

function map_and_update_face_quadrature!(facequads, s, cellid, mergemapper, mapid)
    nfaces = number_of_faces_per_cell(facequads)
    for faceid = 1:nfaces
        quad = facequads[s, faceid, cellid]
        newquad = map_quadrature(quad, mergemapper, mapid)
        update_face_quadrature!(facequads, s, faceid, cellid, newquad)
    end
end

function solution_cell_id(mergedwithcell,s,cellid)
    row = cell_sign_to_row(s)
    return mergedwithcell[row,cellid]
end
