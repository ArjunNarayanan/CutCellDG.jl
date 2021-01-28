struct InterfaceQuadratures
    quads::Any
    normals::Any
    celltoquad::Any
    ncells::Any
    function InterfaceQuadratures(quads, normals, celltoquad)
        ncells = length(celltoquad)
        nphase, nquads = size(quads)

        @assert nphase == 2
        @assert length(normals) == nquads
        @assert all(celltoquad .>= 0)
        @assert all(celltoquad .<= nquads)

        new(quads, normals, celltoquad, ncells)
    end
end

function InterfaceQuadratures(
    cutmesh,
    levelset,
    levelsetcoeffs,
    numqp,
)

    numcells = number_of_cells(cutmesh)

    xL, xR = [-1.0, -1.0], [1.0, 1.0]
    invjac = inverse_jacobian(cutmesh)

    hasinterface = [cell_sign(cutmesh,cellid) == 0 for cellid in 1:numcells]
    numinterfaces = count(hasinterface)
    quads = Matrix(undef, 2, numinterfaces)
    normals = Vector(undef, numinterfaces)
    celltoquad = zeros(Int, numcells)

    interfacecellids = findall(hasinterface)
    counter = 1
    for cellid in interfacecellids
        nodeids = nodal_connectivity(background_mesh(cutmesh),cellid)
        update!(levelset, levelsetcoeffs[nodeids])

        try
            squad = surface_quadrature(levelset, xL, xR, numqp, numsplits = 2)
            n = levelset_normals(levelset, squad.points, invjac)
            quads[1, counter] = squad
            quads[2, counter] = squad
            normals[counter] = n
        catch e
            squad = surface_quadrature(levelset,xL,xR,numqp,numsplits = 3)
            n = levelset_normals(levelset, squad.points, invjac)
            quads[1, counter] = squad
            quads[2, counter] = squad
            normals[counter] = n
        end

        celltoquad[cellid] = counter
        counter += 1
    end
    return InterfaceQuadratures(quads, normals, celltoquad)
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

function update_interface_quadrature!(interfacequads::InterfaceQuadratures, s, cellid, quad)
    row = cell_sign_to_row(s)
    idx = interfacequads.celltoquad[cellid]
    interfacequads.quads[row, idx] = quad
end
