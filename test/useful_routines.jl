using ImplicitDomainQuadrature
import Base.==, Base.≈

function allapprox(v1, v2)
    return all(v1 .≈ v2)
end

function allapprox(v1, v2, atol)
    return length(v1) == length(v2) &&
           all([isapprox(v1[i], v2[i], atol = atol) for i = 1:length(v1)])
end

function allequal(v1, v2)
    return all(v1 .== v2)
end

function Base.isequal(c1::CutCellDG.CellMap, c2::CutCellDG.CellMap)
    return allequal(c1.yL, c2.yL) && allequal(c1.yR, c2.yR)
end

function ==(c1::CutCellDG.CellMap, c2::CutCellDG.CellMap)
    return isequal(c1, c2)
end

function plane_distance_function(coords, normal, x0)
    return (coords .- x0)' * normal
end

function closest_point_on_plane(querypoints,normal,x0)
    normalcomp = vec(normal'*(querypoints .- x0))
    disp = hcat([c*normal for c in normalcomp]...)
    return querypoints - disp
end

function circle_distance_function(coords, center, radius)
    difference = (coords .- center) .^ 2
    distance = radius .- sqrt.(vec(mapslices(sum, difference, dims = 1)))
    return distance
end

function closest_point_on_arc(querypoints,center,radius)
    relpos = querypoints .- center
    cqpoints = relpos[1,:] + im*relpos[2,:]
    theta = angle.(cqpoints)
    v = vcat(cos.(theta)',sin.(theta)')
    return center .+ radius*v
end

function corner_distance_function(x::V, xc) where {V<:AbstractVector}
    v = xc - x
    if all(x .<= xc)
        minimum(v)
    elseif all(x .> xc)
        return -sqrt(v' * v)
    elseif x[2] > xc[2]
        return v[2]
    else
        return v[1]
    end
end

function corner_distance_function(points::M, xc) where {M<:AbstractMatrix}
    return vec(
        mapslices(x -> corner_distance_function(x, xc), points, dims = 1),
    )
end

function ≈(q1::QuadratureRule, q2::QuadratureRule)
    flag = allapprox(q1.points, q2.points) && allapprox(q1.weights, q2.weights)
end

function required_quadrature_order(polyorder)
    ceil(Int, 0.5 * (2polyorder + 1))
end

function add_cell_error_squared!(
    err,
    interpolater,
    exactsolution,
    cellmap,
    quad,
)
    detjac = CutCellDG.determinant_jacobian(cellmap)
    for (p, w) in quad
        numsol = interpolater(p)
        exsol = exactsolution(cellmap(p))
        err .+= (numsol - exsol) .^ 2 * detjac * w
    end
end

function mesh_L2_error(nodalsolutions, exactsolution, basis, cellquads, mesh)
    ndofs = size(nodalsolutions)[1]
    err = zeros(ndofs)
    interpolater = InterpolatingPolynomial(ndofs, basis)
    ncells = CutCellDG.number_of_cells(mesh)
    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)
        if cellsign == +1 || cellsign == 0
            cellmap = CutCellDG.cell_map(mesh, +1, cellid)
            nodeids = CutCellDG.nodal_connectivity(mesh, +1, cellid)
            elementsolution = nodalsolutions[:, nodeids]
            update!(interpolater, elementsolution)
            quad = cellquads[+1, cellid]
            add_cell_error_squared!(
                err,
                interpolater,
                exactsolution,
                cellmap,
                quad,
            )
        end
        if cellsign == -1 || cellsign == 0
            cellmap = CutCellDG.cell_map(mesh, -1, cellid)
            nodeids = CutCellDG.nodal_connectivity(mesh, -1, cellid)
            elementsolution = nodalsolutions[:, nodeids]
            update!(interpolater, elementsolution)
            quad = cellquads[-1, cellid]
            add_cell_error_squared!(
                err,
                interpolater,
                exactsolution,
                cellmap,
                quad,
            )
        end
    end
    return sqrt.(err)
end

function cellwise_L2_error(
    nodalsolutions,
    exactsolution,
    basis,
    cellquads,
    mesh,
)

    ndofs = size(nodalsolutions)[1]
    interpolater = InterpolatingPolynomial(ndofs, basis)
    ncells = CutCellDG.number_of_cells(mesh)
    err = zeros(ndofs, 2, ncells)

    for cellid = 1:ncells
        cellsign = CutCellDG.cell_sign(mesh, cellid)
        CutCellDG.check_cellsign(cellsign)
        if cellsign == +1 || cellsign == 0
            cellmap = CutCellDG.cell_map(mesh, +1, cellid)
            nodeids = CutCellDG.nodal_connectivity(mesh, +1, cellid)
            elementsolution = nodalsolutions[:, nodeids]
            update!(interpolater, elementsolution)
            quad = cellquads[+1, cellid]
            cellerr = zeros(ndofs)
            add_cell_error_squared!(
                cellerr,
                interpolater,
                exactsolution,
                cellmap,
                quad,
            )
            err[:, 1, cellid] = cellerr
        end
        if cellsign == -1 || cellsign == 0
            cellmap = CutCellDG.cell_map(mesh, -1, cellid)
            nodeids = CutCellDG.nodal_connectivity(mesh, -1, cellid)
            elementsolution = nodalsolutions[:, nodeids]
            update!(interpolater, elementsolution)
            quad = cellquads[-1, cellid]
            cellerr = zeros(ndofs)
            add_cell_error_squared!(
                cellerr,
                interpolater,
                exactsolution,
                cellmap,
                quad,
            )
            err[:, 2, cellid] = cellerr
        end
    end
    return sqrt.(err)
end

function average(v)
    return sum(v)/length(v)
end

function convergence_rate(dx, err)
    return diff(log.(err)) ./ diff(log.(dx))
end

function add_cell_norm_squared!(vals, func, cellmap, quad)
    detjac = CutCellDG.determinant_jacobian(cellmap)
    for (p, w) in quad
        v = func(cellmap(p))
        vals .+= v .^ 2 * detjac * w
    end
end

function integral_norm_on_mesh(func, cellquads, mesh, ndofs)
    vals = zeros(ndofs)
    ncells = CutCellDG.number_of_cells(mesh)
    for cellid = 1:ncells
        s = CutCellDG.cell_sign(mesh, cellid)
        @assert s == -1 || s == 0 || s == 1
        if s == 1 || s == 0
            cellmap = CutCellDG.cell_map(mesh, +1, cellid)
            pquad = cellquads[+1, cellid]
            add_cell_norm_squared!(vals, func, cellmap, pquad)
        end
        if s == -1 || s == 0
            cellmap = CutCellDG.cell_map(mesh, -1, cellid)
            nquad = cellquads[-1, cellid]
            add_cell_norm_squared!(vals, func, cellmap, nquad)
        end
    end
    return sqrt.(vals)
end

function add_interface_error_squared!(
    err,
    interpolater,
    exactsolution,
    cellmap,
    quad,
    facescale,
)
    @assert length(quad) == length(facescale)
    for (i, (p, w)) in enumerate(quad)
        numsol = interpolater(p)
        exsol = exactsolution(cellmap(p))
        err .+= (numsol - exsol) .^ 2 * facescale[i] * w
    end
end

function interface_L2_error(
    nodalsolutions,
    exactsolution,
    levelsetsign,
    basis,
    interfacequads,
    mesh,
)
    ndofs = size(nodalsolutions)[1]
    err = zeros(ndofs)
    interpolater = InterpolatingPolynomial(ndofs, basis)

    ncells = CutCellDG.number_of_cells(mesh)

    for cellid = 1:ncells
        if CutCellDG.cell_sign(mesh, cellid) == 0
            quad = interfacequads[levelsetsign, cellid]
            cellmap = CutCellDG.cell_map(mesh, levelsetsign, cellid)
            scalearea = CutCellDG.interface_scale_areas(interfacequads, cellid)

            nodeids = CutCellDG.nodal_connectivity(mesh, levelsetsign, cellid)
            elementsolution = nodalsolutions[:, nodeids]
            update!(interpolater, elementsolution)

            add_interface_error_squared!(
                err,
                interpolater,
                exactsolution,
                cellmap,
                quad,
                scalearea,
            )
        end
    end
    return sqrt.(err)
end

function add_interface_norm_squared!(vals, func, cellmap, quad, facescale)
    @assert length(quad) == length(facescale)
    for (i, (p, w)) in enumerate(quad)
        v = func(cellmap(p))
        vals .+= v .^ 2 * facescale[i] * w
    end
end

function integral_norm_on_interface(
    func,
    interfacequads,
    levelsetsign,
    mesh,
    ndofs,
)
    vals = zeros(ndofs)
    ncells = CutCellDG.number_of_cells(mesh)

    for cellid = 1:ncells
        if CutCellDG.cell_sign(mesh, cellid) == 0
            cellmap = CutCellDG.cell_map(mesh, levelsetsign, cellid)
            quad = interfacequads[levelsetsign, cellid]
            facescale = CutCellDG.interface_scale_areas(interfacequads, cellid)

            add_interface_norm_squared!(vals, func, cellmap, quad, facescale)
        end
    end
    return sqrt.(vals)
end

function update_maxnorm_error!(
    globalerror,
    interpolater,
    exactsolution,
    cellmap,
    quad,
)
    ndofs = length(globalerror)
    for (p, w) in quad
        numsol = interpolater(p)
        exsol = exactsolution(cellmap(p))
        pointerror = abs.(numsol - exsol)
        for i = 1:ndofs
            globalerror[i] = max(globalerror[i], pointerror[i])
        end
    end
end

function interface_maxnorm_error(
    nodalsolutions,
    exactsolution,
    levelsetsign,
    basis,
    interfacequads,
    mesh,
)
    ndofs = size(nodalsolutions)[1]
    err = zeros(ndofs)
    interpolater = InterpolatingPolynomial(ndofs, basis)

    ncells = CutCellDG.number_of_cells(mesh)

    for cellid = 1:ncells
        if CutCellDG.cell_sign(mesh, cellid) == 0
            quad = interfacequads[levelsetsign, cellid]
            cellmap = CutCellDG.cell_map(mesh, levelsetsign, cellid)

            nodeids = CutCellDG.nodal_connectivity(mesh, levelsetsign, cellid)
            elementsolution = nodalsolutions[:, nodeids]
            update!(interpolater, elementsolution)

            update_maxnorm_error!(
                err,
                interpolater,
                exactsolution,
                cellmap,
                quad,
            )
        end
    end
    return err
end


function update_maxnorm!(vals, func, cellmap, quad)
    ndofs = length(vals)
    for (p, w) in quad
        v = abs.(func(cellmap(p)))
        for i = 1:ndofs
            vals[i] = max(vals[i], v[i])
        end
    end
end

function maxnorm_on_interface(func, interfacequads, levelsetsign, mesh, ndofs)
    vals = zeros(ndofs)
    ncells = CutCellDG.number_of_cells(mesh)

    for cellid = 1:ncells
        if CutCellDG.cell_sign(mesh, cellid) == 0
            cellmap = CutCellDG.cell_map(mesh, levelsetsign, cellid)
            quad = interfacequads[levelsetsign, cellid]

            update_maxnorm!(vals, func, cellmap, quad)
        end
    end
    return vals
end

function uniform_mesh_L2_error(nodalsolutions, exactsolution, basis, quad, mesh)
    ndofs, nnodes = size(nodalsolutions)
    err = zeros(ndofs)
    interpolater = InterpolatingPolynomial(ndofs, basis)
    ncells = CutCellDG.number_of_cells(mesh)

    for cellid = 1:ncells
        cellmap = CutCellDG.cell_map(mesh, cellid)
        nodeids = CutCellDG.nodal_connectivity(mesh, cellid)
        elementsolution = nodalsolutions[:, nodeids]
        update!(interpolater, elementsolution)
        add_cell_error_squared!(err, interpolater, exactsolution, cellmap, quad)
    end
    return sqrt.(err)
end

function integral_norm_on_uniform_mesh(func, quad, mesh, ndofs)
    numcells = CutCellDG.number_of_cells(mesh)
    vals = zeros(ndofs)
    for cellid in 1:numcells
        cellmap = CutCellDG.cell_map(mesh,cellid)
        add_cell_norm_squared!(vals, func, cellmap, quad)
    end
    return sqrt.(vals)
end
