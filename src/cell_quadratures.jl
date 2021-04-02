struct CellQuadratures
    quads::Any
    celltoquad::Any
    ncells::Any
    function CellQuadratures(quads, celltoquad)
        nphase, ncells = size(celltoquad)
        @assert nphase == 2
        @assert all(celltoquad .>= 0)
        @assert all(celltoquad .<= length(quads))
        new(quads, celltoquad, ncells)
    end
end

function CellQuadratures(mesh, levelset, numuniformqp, numcutqp)
    numcells = number_of_cells(mesh)

    tpq = tensor_product_quadrature(2, numuniformqp)

    quads = [tpq]

    celltoquad = zeros(Int, 2, numcells)
    xL, xR = [-1.0, -1.0], [1.0, 1.0]

    for cellid = 1:numcells
        s = cell_sign(mesh, cellid)
        if s == +1
            celltoquad[1, cellid] = 1
        elseif s == -1
            celltoquad[2, cellid] = 1
        elseif s == 0
            nodeids = nodal_connectivity(background_mesh(mesh), cellid)
            load_coefficients!(levelset, cellid)

            try
                pquad = area_quadrature(
                    interpolater(levelset),
                    +1,
                    xL,
                    xR,
                    numcutqp,
                    numsplits = 2,
                )
                push!(quads, pquad)
                celltoquad[1, cellid] = length(quads)

                nquad = area_quadrature(
                    interpolater(levelset),
                    -1,
                    xL,
                    xR,
                    numcutqp,
                    numsplits = 2,
                )
                push!(quads, nquad)
                celltoquad[2, cellid] = length(quads)
            catch e
                pquad = area_quadrature(
                    interpolater(levelset),
                    +1,
                    xL,
                    xR,
                    numcutqp,
                    numsplits = 3,
                )
                push!(quads, pquad)
                celltoquad[1, cellid] = length(quads)

                nquad = area_quadrature(
                    interpolater(levelset),
                    -1,
                    xL,
                    xR,
                    numcutqp,
                    numsplits = 3,
                )
                push!(quads, nquad)
                celltoquad[2, cellid] = length(quads)
            end
        else
            error("Expected cellsign ∈ {-1,0,+1}, got cellsign = $s")
        end
    end
    return CellQuadratures(quads, celltoquad)
end

function CellQuadratures(cutmesh, levelset, numqp)
    return CellQuadratures(cutmesh, levelset, numqp, numqp)
end

function Base.getindex(vquads::CellQuadratures, s, cellid)
    row = cell_sign_to_row(s)
    idx = vquads.celltoquad[row, cellid]
    idx > 0 ||
        error("Cell $cellid, cellsign $s, does not have a cell quadrature")
    return vquads.quads[vquads.celltoquad[row, cellid]]
end

function Base.show(io::IO, cellquads::CellQuadratures)
    ncells = cellquads.ncells
    nuniquequads = length(cellquads.quads)
    str = "CellQuadratures\n\tNum. Cells: $ncells\n\tNum. Unique Quadratures: $nuniquequads"
    print(io, str)
end

function uniform_cell_quadrature(vquads::CellQuadratures)
    return vquads.quads[1]
end

function update_cell_quadrature!(cellquads::CellQuadratures, s, cellid, quad)
    row = cell_sign_to_row(s)
    idx = cellquads.celltoquad[row, cellid]
    cellquads.quads[idx] = quad
end
