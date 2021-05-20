struct InterfaceQuadratures
    quads::Any
    normals::Any
    tangents::Any
    scaleareas::Any
    celltoquad::Any
    ncells::Any
    totalnumqps::Any
    function InterfaceQuadratures(
        quads,
        normals,
        tangents,
        scaleareas,
        celltoquad,
    )
        ncells = length(celltoquad)
        nphase, nquads = size(quads)

        interfacenumqps = length.(quads)
        interfacenumqps = nquads == 0 ? [0, 0] : sum(interfacenumqps, dims = 2)
        @assert interfacenumqps[1] == interfacenumqps[2]
        totalnumqps = interfacenumqps[1]

        @assert nphase == 2
        @assert length(normals) == nquads
        @assert length(tangents) == nquads
        @assert length(scaleareas) == nquads
        @assert all(celltoquad .>= 0)
        @assert all(celltoquad .<= nquads)

        new(
            quads,
            normals,
            tangents,
            scaleareas,
            celltoquad,
            ncells,
            totalnumqps,
        )
    end
end

function InterfaceQuadratures(cutmesh, levelset, numqp)
    polyorder = order(levelset)
    dim = dimension(cutmesh)
    interpgrad = InterpolatingPolynomial(dim, dim, polyorder)
    numcells = number_of_cells(cutmesh)

    xL, xR = [-1.0, -1.0], [1.0, 1.0]
    invjac = inverse_jacobian(cutmesh)

    hasinterface = [cell_sign(cutmesh, cellid) == 0 for cellid = 1:numcells]
    numinterfaces = count(hasinterface)
    quads = Matrix(undef, 2, numinterfaces)
    normals = Vector(undef, numinterfaces)
    tangents = Vector(undef, numinterfaces)
    scaleareas = Vector(undef, numinterfaces)
    celltoquad = zeros(Int, numcells)
    invjac = inverse_jacobian(cutmesh)

    interfacecellids = findall(hasinterface)
    counter = 1
    for cellid in interfacecellids
        load_coefficients!(levelset, cellid)
        update_interpolating_gradient!(interpgrad,interpolater(levelset))
        cellmap = cell_map(cutmesh, cellid)

        try
            update_interface_quadrature!(
                quads,
                normals,
                tangents,
                scaleareas,
                counter,
                interpolater(levelset),
                interpgrad,
                xL,
                xR,
                numqp,
                invjac,
                2,
            )
        catch e
            update_interface_quadrature!(
                quads,
                normals,
                tangents,
                scaleareas,
                counter,
                interpolater(levelset),
                interpgrad,
                xL,
                xR,
                numqp,
                invjac,
                3,
            )
        end

        celltoquad[cellid] = counter
        counter += 1
    end
    return InterfaceQuadratures(
        quads,
        normals,
        tangents,
        scaleareas,
        celltoquad,
    )
end

function total_number_of_quadrature_points(interfacequads::InterfaceQuadratures)
    return interfacequads.totalnumqps
end

function update_interface_quadrature!(
    quads,
    normals,
    tangents,
    scaleareas,
    counter,
    levelset,
    levelsetgrad,
    xL,
    xR,
    numqp,
    invjac,
    numsplits,
)
    squad = surface_quadrature(
        levelset,
        levelsetgrad,
        xL,
        xR,
        numqp,
        numsplits = numsplits,
    )
    n = levelset_normals(levelset, points(squad), invjac)
    t = rotate_90(n)
    s = scale_area(t, invjac)

    quads[1, counter] = squad
    quads[2, counter] = squad
    normals[counter] = n
    tangents[counter] = t
    scaleareas[counter] = s
end

function Base.getindex(iquads::InterfaceQuadratures, s, cellid)
    row = cell_sign_to_row(s)
    idx = iquads.celltoquad[cellid]
    idx > 0 || error("Cell $cellid does not have an interface quadrature rule")
    return iquads.quads[row, idx]
end

function Base.show(io::IO, interfacequads::InterfaceQuadratures)
    ncells = interfacequads.ncells
    numinterfaces = length(interfacequads.normals)
    str = "InterfaceQuadratures\n\tNum. Cells: $ncells\n\tNum. Interfaces: $numinterfaces"
    print(io, str)
end

function interface_normals(iquads::InterfaceQuadratures, cellid)
    idx = iquads.celltoquad[cellid]
    idx > 0 || error("Cell $cellid does not have an interface quadrature rule")
    return iquads.normals[idx]
end

function interface_scale_areas(iquads::InterfaceQuadratures, cellid)
    idx = iquads.celltoquad[cellid]
    idx > 0 || error("Cell $cellid does not have an interface quadrature rule")
    return iquads.scaleareas[idx]
end

function interface_tangents(iquads::InterfaceQuadratures, cellid)
    idx = iquads.celltoquad[cellid]
    idx > 0 || error("Cell $cellid does not have an interface quadrature rule")
    return iquads.tangents[idx]
end

function update_interface_quadrature!(
    interfacequads::InterfaceQuadratures,
    s,
    cellid,
    quad,
)
    row = cell_sign_to_row(s)
    idx = interfacequads.celltoquad[cellid]
    interfacequads.quads[row, idx] = quad
end

function collect_interface_normals(interfacequads, mesh)

    ncells = number_of_cells(mesh)
    cellsign = [cell_sign(mesh, cellid) for cellid = 1:ncells]
    cellids = findall(cellsign .== 0)

    return collect_interface_normals(interfacequads, mesh, cellids)
end

function collect_interface_normals(interfacequads, mesh, cellids)

    totalnumqps = sum([length(interfacequads[1, cellid]) for cellid in cellids])
    interfacenormals = zeros(2, totalnumqps)
    start = 1

    for cellid in cellids
        normals = interface_normals(interfacequads, cellid)
        numqps = size(normals)[2]

        stop = start + numqps - 1

        interfacenormals[:, start:stop] = normals
        start = stop + 1
    end
    return interfacenormals
end

function collect_interface_quadrature_points(
    interfacequads,
    cellsign,
    mesh,
    cellids,
)
    totalnumqps = sum([length(interfacequads[1, cellid]) for cellid in cellids])
    quadraturepoints = zeros(2, totalnumqps)
    quadraturecellids = zeros(Int, totalnumqps)
    start = 1

    for cellid in cellids
        quad = interfacequads[cellsign, cellid]
        qps = points(quad)
        numqps = size(qps)[2]

        stop = start + numqps - 1

        quadraturepoints[:, start:stop] = qps
        quadraturecellids[start:stop] = repeat([cellid], numqps)
        start = stop + 1
    end
    return quadraturepoints, quadraturecellids
end

function collect_interface_quadrature_points(interfacequads, levelsetsign, mesh)

    ncells = number_of_cells(mesh)
    cellsign = [cell_sign(mesh, cellid) for cellid = 1:ncells]
    cellids = findall(cellsign .== 0)
    qps, qpcellids = collect_interface_quadrature_points(
        interfacequads,
        levelsetsign,
        mesh,
        cellids,
    )

    return qps, qpcellids
end
