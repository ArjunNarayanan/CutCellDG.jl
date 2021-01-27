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

function CellQuadratures(
    mesh,
    levelset,
    levelsetcoeffs,
    numuniformqp,
    numcutqp,
)
    numcells = number_of_cells(mesh)

    tpq = tensor_product_quadrature(2, numuniformqp)

    quads = [tpq]

    celltoquad = zeros(Int, 2, numcells)
    xL, xR = [-1.0, -1.0], [1.0, 1.0]

    for cellid = 1:numcells
        s = cell_sign(mesh,cellid)
        if s == +1
            celltoquad[1, cellid] = 1
        elseif s == -1
            celltoquad[2, cellid] = 1
        elseif s == 0
            nodeids = nodal_connectivity(background_mesh(mesh),cellid)
            update!(levelset, levelsetcoeffs[nodeids])

            try
                pquad = area_quadrature(levelset, +1, xL, xR, numcutqp, numsplits = 2)
                push!(quads, pquad)
                celltoquad[1, cellid] = length(quads)

                nquad = area_quadrature(levelset, -1, xL, xR, numcutqp, numsplits = 2)
                push!(quads, nquad)
                celltoquad[2, cellid] = length(quads)
            catch e
                pquad = area_quadrature(levelset, +1, xL, xR, numcutqp, numsplits = 3)
                push!(quads, pquad)
                celltoquad[1, cellid] = length(quads)

                nquad = area_quadrature(levelset, -1, xL, xR, numcutqp, numsplits = 3)
                push!(quads, nquad)
                celltoquad[2, cellid] = length(quads)
            end
        else
            error("Expected cellsign ∈ {-1,0,+1}, got cellsign = $s")
        end
    end
    return CellQuadratures(quads, celltoquad)
end
