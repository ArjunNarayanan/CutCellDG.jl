struct BoundaryPaddedMesh
    mesh::Any
    gridsize::Any
    nfmside::Any
    numghostlayers::Any
    bottomghostcoords::Any
    rightghostcoords::Any
    topghostcoords::Any
    leftghostcoords::Any
end

function Base.show(io::IO, mesh::BoundaryPaddedMesh)
    dim = dimension(mesh)
    numghostlayers = number_of_ghost_layers(mesh)
    gridsize = grid_size(mesh)
    nfmside = nodes_per_mesh_side(mesh)

    str =
        "BoundaryPaddedMesh\n\tDimension : $dim\n\tGrid Size : $gridsize" *
        "\n\tNodes per side : $nfmside" *
        "\n\tNum. Ghost Layers : $numghostlayers"
    print(io, str)
end

function BoundaryPaddedMesh(mesh, numghostlayers)
    x0 = reference_corner(mesh)
    nfmside = nodes_per_mesh_side(mesh)
    meshwidths = mesh_widths(mesh)

    gridsize = meshwidths ./ (nfmside .- 1)

    bottomghostcoords = bottom_ghost_coordinates(
        x0,
        meshwidths[1],
        nfmside[1],
        gridsize[2],
        numghostlayers,
    )
    rightghostcoords = right_ghost_coordinates(
        x0,
        meshwidths,
        nfmside[2],
        gridsize[1],
        numghostlayers,
    )
    topghostcoords = top_ghost_coordinates(
        x0,
        meshwidths,
        nfmside[1],
        gridsize[2],
        numghostlayers,
    )
    leftghostcoords = left_ghost_coordinates(
        x0,
        meshwidths[2],
        nfmside[2],
        gridsize[1],
        numghostlayers,
    )

    return BoundaryPaddedMesh(
        mesh,
        gridsize,
        nfmside,
        numghostlayers,
        bottomghostcoords,
        rightghostcoords,
        topghostcoords,
        leftghostcoords,
    )
end

function background_mesh(mesh::BoundaryPaddedMesh)
    return mesh.mesh
end

function dimension(mesh::BoundaryPaddedMesh)
    return dimension(background_mesh(mesh))
end

function nodes_per_mesh_side(paddedmesh::BoundaryPaddedMesh)
    return paddedmesh.nfmside
end

function grid_size(mesh::BoundaryPaddedMesh)
    return mesh.gridsize
end

function number_of_ghost_layers(mesh::BoundaryPaddedMesh)
    return mesh.numghostlayers
end

function bottom_ghost_coordinates(mesh::BoundaryPaddedMesh)
    return mesh.bottomghostcoords
end

function right_ghost_coordinates(mesh::BoundaryPaddedMesh)
    return mesh.rightghostcoords
end

function top_ghost_coordinates(mesh::BoundaryPaddedMesh)
    return mesh.topghostcoords
end

function left_ghost_coordinates(mesh::BoundaryPaddedMesh)
    return mesh.leftghostcoords
end

function uniform_grid_coordinates(x0, widths, npts)
    xrange = range(x0[1], stop = x0[1] + widths[1], length = npts[1])
    yrange = range(x0[2], stop = x0[2] + widths[2], length = npts[2])

    xcoords = repeat(xrange, inner = npts[2])
    ycoords = repeat(yrange, outer = npts[1])

    return vcat(xcoords', ycoords')
end

function bottom_ghost_coordinates(x0, xwidth, npts, dy, numlayers)

    return uniform_grid_coordinates(
        x0 - [0.0, numlayers * dy],
        [xwidth, (numlayers - 1) * dy],
        [npts, numlayers],
    )
end

function right_ghost_coordinates(x0, widths, npts, dx, numlayers)

    return uniform_grid_coordinates(
        x0 + [widths[1] + dx, 0.0],
        [(numlayers - 1) * dx, widths[2]],
        [numlayers, npts],
    )
end

function top_ghost_coordinates(x0, widths, npts, dy, numlayers)

    return uniform_grid_coordinates(
        x0 + [0.0, widths[2] + dy],
        [widths[1], (numlayers - 1) * dy],
        [npts, numlayers],
    )
end

function left_ghost_coordinates(x0, ywidth, npts, dx, numlayers)

    return uniform_grid_coordinates(
        x0 - [numlayers * dx, 0.0],
        [(numlayers - 1) * dx, ywidth],
        [numlayers, npts],
    )
end

struct BoundaryPaddedLevelSet
    interiordist::Any
    bottomghostdist::Any
    rightghostdist::Any
    topghostdist::Any
    leftghostdist::Any
    paddedmesh::Any
end

function BoundaryPaddedLevelSet(
    paddedmesh::BoundaryPaddedMesh,
    spatialseedpoints,
    seedcellids,
    levelset,
    tol,
    boundingradius,
)

    bottomghostdist = distance_to_zero_levelset(
        bottom_ghost_coordinates(paddedmesh),
        spatialseedpoints,
        seedcellids,
        levelset,
        tol,
        boundingradius,
    )
    rightghostdist = distance_to_zero_levelset(
        right_ghost_coordinates(paddedmesh),
        spatialseedpoints,
        seedcellids,
        levelset,
        tol,
        boundingradius,
    )
    topghostdist = distance_to_zero_levelset(
        top_ghost_coordinates(paddedmesh),
        spatialseedpoints,
        seedcellids,
        levelset,
        tol,
        boundingradius,
    )
    leftghostdist = distance_to_zero_levelset(
        left_ghost_coordinates(paddedmesh),
        spatialseedpoints,
        seedcellids,
        levelset,
        tol,
        boundingradius,
    )

    return BoundaryPaddedLevelSet(
        coefficients(levelset),
        bottomghostdist,
        rightghostdist,
        topghostdist,
        leftghostdist,
        paddedmesh,
    )
end

function background_mesh(p::BoundaryPaddedLevelSet)
    return p.paddedmesh
end

function interior_values(paddedlevelset::BoundaryPaddedLevelSet)
    return paddedlevelset.interiordist
end

function left_ghost_values(paddedlevelset::BoundaryPaddedLevelSet)
    return paddedlevelset.leftghostdist
end

function right_ghost_values(paddedlevelset::BoundaryPaddedLevelSet)
    return paddedlevelset.rightghostdist
end

function bottom_ghost_values(paddedlevelset::BoundaryPaddedLevelSet)
    return paddedlevelset.bottomghostdist
end

function top_ghost_values(paddedlevelset::BoundaryPaddedLevelSet)
    return paddedlevelset.topghostdist
end

function first_order_horizontal_backward_difference(paddedlevelset)
    interior = interior_values(paddedlevelset)
    left = left_ghost_values(paddedlevelset)

    cols, rows = nodes_per_mesh_side(background_mesh(paddedlevelset))
    dx = grid_size(background_mesh(paddedlevelset))[1]

    npts = length(interior)
    derivative = zeros(npts)

    derivative[1:rows] = (interior[1:rows] - left) / dx

    for col = 2:cols
        prevstart = (col - 2) * rows + 1
        prevstop = prevstart + rows - 1

        start = (col - 1) * rows + 1
        stop = start + rows - 1

        derivative[start:stop] =
            (interior[start:stop] - interior[prevstart:prevstop]) / dx
    end

    return derivative
end

function first_order_horizontal_forward_difference(paddedlevelset)

    interior = interior_values(paddedlevelset)
    right = right_ghost_values(paddedlevelset)
    cols, rows = nodes_per_mesh_side(background_mesh(paddedlevelset))
    dx = grid_size(background_mesh(paddedlevelset))[1]

    npts = length(interior)
    derivative = zeros(npts)

    for col = 1:cols-1
        start = (col - 1) * rows + 1
        stop = start + rows - 1
        nextstart = stop + 1
        nextstop = nextstart + rows - 1

        derivative[start:stop] =
            (interior[nextstart:nextstop] - interior[start:stop]) / dx
    end

    start = (cols - 1) * rows + 1
    stop = start + rows - 1
    derivative[start:stop] = (right - interior[start:stop]) / dx

    return derivative
end

function first_order_vertical_backward_difference(paddedlevelset)

    interior = interior_values(paddedlevelset)
    bottom = bottom_ghost_values(paddedlevelset)
    cols, rows = nodes_per_mesh_side(background_mesh(paddedlevelset))
    dy = grid_size(background_mesh(paddedlevelset))[2]
    numghostlayers = number_of_ghost_layers(background_mesh(paddedlevelset))

    npts = length(interior)
    derivative = zeros(npts)

    interioridx = range(1, step = rows, length = cols)
    ghostidx = range(numghostlayers, step = numghostlayers, length = cols)

    derivative[interioridx] = (interior[interioridx] - bottom[ghostidx]) / dy

    for row = 2:rows
        interioridx = range(row, step = rows, length = cols)
        previnterioridx = range(row - 1, step = rows, length = cols)

        derivative[interioridx] =
            (interior[interioridx] - interior[previnterioridx]) / dy
    end

    return derivative
end

function first_order_vertical_forward_difference(paddedlevelset)
    interior = interior_values(paddedlevelset)
    top = top_ghost_values(paddedlevelset)
    cols, rows = nodes_per_mesh_side(background_mesh(paddedlevelset))
    dy = grid_size(background_mesh(paddedlevelset))[2]
    numghostlayers = number_of_ghost_layers(background_mesh(paddedlevelset))

    npts = length(interior)
    derivative = zeros(npts)

    for row = 1:rows-1
        interioridx = range(row, step = rows, length = cols)
        nextinterioridx = range(row + 1, step = rows, length = cols)

        derivative[interioridx] =
            (interior[nextinterioridx] - interior[interioridx]) / dy
    end

    interioridx = range(rows, step = rows, length = cols)
    ghostidx = range(1, step = numghostlayers, length = cols)

    derivative[interioridx] = (top[ghostidx] - interior[interioridx]) / dy

    return derivative
end

function first_order_nabla(u1, u2, v1, v2)
    return sqrt.(
        (max.(u1, 0.0)) .^ 2 +
        (min.(u2, 0.0)) .^ 2 +
        (max.(v1, 0.0)) .^ 2 +
        (min.(v2, 0.0)) .^ 2,
    )
end

function step_first_order_levelset(paddedlevelset, speed, dt)
    Dmx = first_order_horizontal_backward_difference(paddedlevelset)
    Dpx = first_order_horizontal_forward_difference(paddedlevelset)
    Dmy = first_order_vertical_backward_difference(paddedlevelset)
    Dpy = first_order_vertical_forward_difference(paddedlevelset)

    delplus = first_order_nabla(Dmx, Dpx, Dmy, Dpy)
    delminus = first_order_nabla(Dpx, Dmx, Dpy, Dmy)

    return paddedlevelset.interiordist -
           dt * (max.(speed, 0.0) .* delplus + min.(speed, 0.0) .* delminus)
end
