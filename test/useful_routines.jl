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

function circle_distance_function(coords, center, radius)
    difference = (coords .- center) .^ 2
    distance = radius .- sqrt.(vec(mapslices(sum, difference, dims = 1)))
    return distance
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

function cellwise_L2_error(nodalsolutions, exactsolution, basis, cellquads, mesh)

    ndofs = size(nodalsolutions)[1]
    interpolater = InterpolatingPolynomial(ndofs, basis)
    ncells = CutCellDG.number_of_cells(mesh)
    err = zeros(ndofs,2,ncells)

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
            err[:,1,cellid] = cellerr
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
            err[:,2,cellid] = cellerr
        end
    end
    return sqrt.(err)
end

function convergence_rate(dx, err)
    return diff(log.(err)) ./ diff(log.(dx))
end
