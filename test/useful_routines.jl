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
