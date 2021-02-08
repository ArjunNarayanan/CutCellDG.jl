struct InterfaceQuadratures
    quads::Any
    normals::Any
    tangents
    scaleareas::Any
    celltoquad::Any
    ncells::Any
    totalnumqps
    function InterfaceQuadratures(quads, normals, tangents, scaleareas, celltoquad)
        ncells = length(celltoquad)
        nphase, nquads = size(quads)

        interfacenumqps = length.(quads)
        interfacenumqps = nquads == 0 ? [0,0] : sum(interfacenumqps,dims=2)
        @assert interfacenumqps[1] == interfacenumqps[2]
        totalnumqps = interfacenumqps[1]

        @assert nphase == 2
        @assert length(normals) == nquads
        @assert length(tangents) == nquads
        @assert length(scaleareas) == nquads
        @assert all(celltoquad .>= 0)
        @assert all(celltoquad .<= nquads)

        new(quads, normals, tangents, scaleareas, celltoquad, ncells, totalnumqps)
    end
end

function InterfaceQuadratures(cutmesh::CutMesh, levelset, levelsetcoeffs, numqp)

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
        nodeids = nodal_connectivity(background_mesh(cutmesh), cellid)
        update!(levelset, levelsetcoeffs[nodeids])
        cellmap = cell_map(cutmesh, cellid)

        try
            update_interface_quadrature!(
                quads,
                normals,
                tangents,
                scaleareas,
                counter,
                levelset,
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
                levelset,
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
    xL,
    xR,
    numqp,
    invjac,
    numsplits,
)
    squad = surface_quadrature(levelset, xL, xR, numqp, numsplits = numsplits)
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

function collect_interface_normals(interfacequads,mesh)

    totalnumqps = total_number_of_quadrature_points(interfacequads)
    ncells = number_of_cells(mesh)
    interfacenormals = zeros(2,totalnumqps)
    start = 1

    for cellid in 1:ncells
        cellsign = cell_sign(mesh,cellid)
        if cellsign == 0
            normals = interface_normals(interfacequads,cellid)
            numqps = size(normals)[2]

            stop = start + numqps - 1

            interfacenormals[:,start:stop] = normals
            start = stop + 1
        end
    end
    return interfacenormals
end
