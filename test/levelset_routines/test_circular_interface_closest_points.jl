using Test
using LinearAlgebra
using PolynomialBasis
# using Revise
using CutCellDG
include("../useful_routines.jl")

function polar_to_cartesian(radialposition, angularposition)
    x = radialposition .* cos.(angularposition)
    y = radialposition .* sin.(angularposition)
    return vcat(x', y')
end

L, W = 1.0, 1.0
nelmts = [4, 4]
center = [0.5, 0.5]
radius = 0.4
polyorder = 2

basis = TensorProductBasis(2, polyorder)
mesh = CutCellDG.CGMesh([0.0, 0.0], [L, W], nelmts, basis)
levelset = CutCellDG.LevelSet(
    x -> circle_distance_function(x, center, radius),
    mesh,
    basis,
)
cutmesh = CutCellDG.CutMesh(mesh, levelset)

refseedpoints, seedcellids = CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
seedpoints = CutCellDG.map_to_spatial(refseedpoints, seedcellids, mesh)

numquerypts = 100
queryangles = range(0, stop = 2pi, length = numquerypts)
queryradius = 0.1 .+ abs.(sin.(queryangles))
querypoints = center .+ polar_to_cartesian(queryradius, queryangles)

dx = CutCellDG.element_size(mesh)
tol = 1e3eps()
boundingradius = 1.5 * maximum(dx)
closestpoints, closestcellids, flags =
    CutCellDG.closest_points_on_zero_levelset(
        querypoints,
        seedpoints,
        seedcellids,
        levelset,
        tol,
        boundingradius,
    )


testcp = closest_point_on_arc(querypoints, center, radius)
err = norm(testcp - closestpoints, Inf)
# err = vec(mapslices(x->norm(x,Inf),closestpoints - testcp,dims=1))
# maxidx = argmax(err)
# maxerr = err[maxidx]

@test all(flags)
@test err < 0.05







# xq = querypoints[:,1]
# xcp = closestpoints[:,1]
# ecp = testcp[:,1]
#
# nodalcoordinates = CutCellDG.nodal_coordinates(mesh)
# levelsetcoeffs = CutCellDG.coefficients(levelset)
# using PyPlot
# fig, ax = PyPlot.subplots()
# ax.tricontour(
#     nodalcoordinates[1, :],
#     nodalcoordinates[2, :],
#     levelsetcoeffs,
#     [0.0],
# )
# ax.scatter([xq[1]],[xq[2]])
# ax.scatter([xcp[1]],[xcp[2]])
# ax.scatter([ecp[1]],[ecp[2]])
# ax.grid()
# ax.set_aspect("equal")
# fig