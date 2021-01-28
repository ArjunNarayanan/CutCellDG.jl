struct FaceQuadratures
    quads::Any
    facetoquad::Any
    ncells::Any
    nfaces::Any
    function FaceQuadratures(quads, facetoquad)
        nphase, nfaces, ncells = size(facetoquad)
        @assert nfaces == 4
        @assert nphase == 2
        @assert all(facetoquad .>= 0)
        @assert all(facetoquad .<= length(quads))
        new(quads, facetoquad, ncells, nfaces)
    end
end

function FaceQuadratures(cutmesh, levelset, levelsetcoeffs, numqp)

    ncells = number_of_cells(cutmesh)

    quads = face_quadratures(numqp)
    quad1d = ReferenceQuadratureRule(numqp)
    facetoquad = zeros(Int, 2, 4, ncells)

    for cellid = 1:ncells
        s = cell_sign(cutmesh,cellid)
        if s == +1
            facetoquad[1, :, cellid] .= 1:4
        elseif s == -1
            facetoquad[2, :, cellid] .= 1:4
        elseif s == 0
            nodeids = nodal_connectivity(background_mesh(cutmesh),cellid)
            update!(levelset, levelsetcoeffs[nodeids])

            pquad = face_quadratures(levelset, +1, quad1d)
            idxstart = length(quads) + 1
            append!(quads, pquad)
            facetoquad[1, :, cellid] .= idxstart:(idxstart+3)

            nquad = face_quadratures(levelset, -1, quad1d)
            idxstart = length(quads) + 1
            append!(quads, nquad)
            facetoquad[2, :, cellid] .= idxstart:(idxstart+3)
        else
            error("Expected cellsign ∈ {-1,0,+1}, got cellsign = $s")
        end
    end
    return FaceQuadratures(quads, facetoquad)
end

function Base.getindex(facequads::FaceQuadratures, s, faceid, cellid)
    phaseid = cell_sign_to_row(s)
    return facequads.quads[facequads.facetoquad[phaseid, faceid, cellid]]
end

function uniform_face_quadratures(facequads::FaceQuadratures)
    return facequads.quads[1:facequads.nfaces]
end

function number_of_faces_per_cell(facequads::FaceQuadratures)
    return facequads.nfaces
end

function Base.show(io::IO, facequads::FaceQuadratures)
    ncells = facequads.ncells
    nuniquefacequads = length(facequads.quads)
    str = "FaceQuadratures\n\tNum. Cells: $ncells\n\tNum. Unique Quadratures: $nuniquefacequads"
    print(io, str)
end

function face_quadratures(numqp)
    quad1d = tensor_product_quadrature(1, numqp)
    facequads = [extend_to_face(quad1d, faceid) for faceid = 1:4]
    return facequads
end

function face_quadrature(faceid, levelset, signcondition, quad1d)
    dir, coordval = reference_face(faceid)
    quad = QuadratureRule(ImplicitDomainQuadrature.one_dimensional_quadrature(
        [x -> levelset(extend(x, dir, coordval))],
        [signcondition],
        -1.0,
        +1.0,
        quad1d,
    ))
    return extend_to_face(quad, faceid)
end

function face_quadratures(levelset, signcondition, quad1d)
    quads = [face_quadrature(faceid, levelset, signcondition, quad1d) for faceid = 1:4]
    return quads
end

function extend_to_face(quad::QuadratureRule, faceid)
    extp = extend_to_face(quad.points, faceid)
    return QuadratureRule(extp, quad.weights)
end

function extend_to_face(points, faceid)
    dir, coordval = reference_face(faceid)
    @assert dir == 1 || dir == 2
    flipdir = dir == 1 ? 2 : 1
    return extend([coordval], flipdir, points)
end

function reference_face(faceid)
    if faceid == 1
        return (2, -1.0)
    elseif faceid == 2
        return (1, +1.0)
    elseif faceid == 3
        return (2, +1.0)
    elseif faceid == 4
        return (1, -1.0)
    else
        error("Expected faceid ∈ {1,2,3,4}, got faceid = $faceid")
    end
end

function update_face_quadrature!(facequads::FaceQuadratures, s, faceid, cellid, quad)
    row = cell_sign_to_row(s)
    idx = facequads.facetoquad[row, faceid, cellid]
    @assert idx > 4 "Attempting to modify the uniform cell face quadrature rules"
    facequads.quads[idx] = quad
end
