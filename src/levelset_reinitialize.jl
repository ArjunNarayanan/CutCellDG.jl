function hessian_matrix(poly, x)
    h = hessian(poly, x)
    return [
        h[1] h[2]
        h[2] h[3]
    ]
end

# function seed_zero_levelset_with_interfacequads(interfacequads, mesh)
#
#     ncells = number_of_cells(mesh)
#     cellsign = [cell_sign(mesh, cellid) for cellid = 1:ncells]
#     cellids = findall(cellsign .== 0)
#
#     return seed_zero_levelset_with_interfacequads(interfacequads, mesh, cellids)
# end

# function seed_zero_levelset_with_interfacequads(interfacequads, mesh, cellids)
#
#     totalnumqps = sum([length(interfacequads[1, cellid]) for cellid in cellids])
#     dim = dimension(mesh)
#
#     refseedpoints = zeros(2, dim, totalnumqps)
#     # spatialseedpoints = zeros(2, dim, totalnumqps)
#     seedcellids = zeros(Int, 2, totalnumqps)
#
#     start = 1
#     for cellid in cellids
#         cellsign = cell_sign(mesh, cellid)
#
#         if cellsign == 0
#             numqps = length(interfacequads[1, cellid])
#             stop = start + numqps - 1
#             for s in [+1, -1]
#                 cellmap = cell_map(mesh, s, cellid)
#                 refpoints = points(interfacequads[s, cellid])
#                 spatialpoints = cellmap(refpoints)
#
#                 row = cell_sign_to_row(s)
#
#                 refseedpoints[row, :, start:stop] = refpoints
#                 # spatialseedpoints[row, :, start:stop] = spatialpoints
#
#                 seedcellid = solution_cell_id(mesh, s, cellid)
#                 seedcellids[row, start:stop] = repeat([seedcellid], numqps)
#             end
#             start = stop + 1
#         end
#     end
#     return refseedpoints, seedcellids
#     # return refseedpoints, spatialseedpoints, seedcellids
# end

function project_on_zero_levelset(
    xguess,
    func,
    grad,
    tol,
    r;
    maxiter = 20,
    normtol = 1e-8,
    perturbation = 0.1,
)
    x0 = copy(xguess)
    x1 = copy(x0)
    dim = length(xguess)

    for counter = 1:maxiter
        vf = func(x0)
        gf = grad(x0)

        if norm(gf) < normtol
            x0 = x0 + perturbation * (rand(dim) .- 0.5)
        else
            δ = vf / (gf' * gf) * gf
            normδ = norm(δ)
            if normδ > 0.5r
                δ *= 0.5r / normδ
            end

            x1 = x0 - δ

            if abs(func(x1)) < tol
                flag = norm(x1 - xguess) < r
                return x1, flag
            else
                x0 = x1
            end
        end
    end
    error("Did not converge after $maxiter iterations")
end

function reference_seed_points(n)
    @assert n > 0
    xrange = range(-1.0, stop = 1.0, length = n + 2)
    points = ImplicitDomainQuadrature.tensor_product_points(
        xrange[2:n+1]',
        xrange[2:n+1]',
    )
end

function seed_cell_zero_levelset(xguess, func, grad, tol, boundingradius)
    dim, nump = size(xguess)
    pf = [
        project_on_zero_levelset(xguess[:, i], func, grad, tol, boundingradius) for i = 1:nump
    ]
    flags = [p[2] for p in pf]
    valididx = findall(flags)
    validpoints = [p[1] for p in pf[valididx]]
    return hcat(validpoints...)
end

function seed_zero_levelset(
    nump,
    levelset,
    cutmesh;
    tol = 1e-12,
    boundingradius = 4.0,
)

    refpoints = reference_seed_points(nump)
    refseedpoints = []
    seedcellids = Int[]
    numcells = number_of_cells(cutmesh)

    cellsign = [cell_sign(cutmesh, cellid) for cellid = 1:numcells]
    cellids = findall(cellsign .== 0)

    for cellid in cellids
        load_coefficients!(levelset, cellid)

        xk = seed_cell_zero_levelset(
            refpoints,
            interpolater(levelset),
            x -> vec(gradient(interpolater(levelset), x)),
            tol,
            boundingradius,
        )

        numseedpoints = size(xk)[2]

        append!(seedcellids, repeat([cellid], numseedpoints))
        push!(refseedpoints, xk)
    end

    refseedpoints = hcat(refseedpoints...)
    return refseedpoints, seedcellids
end

function saye_newton_iterate(
    xguess,
    xq,
    func,
    grad,
    hess,
    cellmap,
    tol,
    r;
    maxiter = 20,
    condtol = 1e5eps(),
)
    dim = length(xguess)
    jac = jacobian(cellmap)

    x0 = copy(xguess)
    gp = grad(x0)
    l0 = gp' * ((xq - cellmap(x0)) .* jac) / (gp' * gp)

    x1 = copy(x0)
    l1 = l0

    for counter = 1:maxiter
        vp = func(x0)
        gp = grad(x0)
        hp = hess(x0)

        gf = vcat(((cellmap(x0) - xq) .* jac) + l0 * gp, vp)
        hf = [
            diagm(jac .^ 2)+l0*hp gp
            gp' 0.0
        ]

        if inv(cond(hf)) > condtol
            δ = hf \ gf
            normδx = norm(δ[1:dim])
            if normδx > 0.5r
                δ *= 0.5r / normδx
            end
            x1 = x0 - δ[1:dim]
            l1 = l0 - δ[end]
        else
            error("Chopp method not implemented")
        end

        if norm(x1 - xguess) > r
            error("Did not converge in ball of radius $r")
        elseif norm(x1 - x0) < tol
            return x1
        else
            x0 = x1
            l0 = l1
        end
    end
    error("Did not converge in $maxiter iterations")
end

# function closest_reference_points_on_merged_mesh(
#     querypoints,
#     refseedpoints,
#     spatialseedpoints,
#     seedcellids,
#     levelset,
#     mesh,
#     tol,
#     boundingradius,
# )
#
#     dim, numquerypoints = size(querypoints)
#     refclosestpoints = zeros(2, dim, numquerypoints)
#     refclosestcellids = zeros(Int, 2, numquerypoints)
#
#     tree = KDTree(spatialseedpoints)
#     seedidx, seeddists = nn(tree, querypoints)
#
#     for (idx, sidx) in enumerate(seedidx)
#         for cellsign in [+1, -1]
#             row = cell_sign_to_row(cellsign)
#
#             xguess = refseedpoints[row, :, sidx]
#             xquery = querypoints[:, idx]
#             guesscellid = seedcellids[row, sidx]
#
#             cellmap = cell_map(mesh, cellsign, guesscellid)
#             load_coefficients!(levelset, guesscellid)
#             interpolatingpoly = interpolater(levelset)
#
#             try
#                 refcp = saye_newton_iterate(
#                     xguess,
#                     xquery,
#                     interpolatingpoly,
#                     x -> vec(gradient(interpolatingpoly, x)),
#                     x -> hessian_matrix(interpolatingpoly, x),
#                     cellmap,
#                     tol,
#                     boundingradius,
#                 )
#
#                 refclosestpoints[row, :, idx] = refcp
#                 refclosestcellids[row, idx] = guesscellid
#             catch e
#                 println("Newton iteration failed to converge")
#                 println("\tQuery point index = $idx")
#                 println("\tSeed point index = $sidx")
#                 println("\tCellid = $guesscellid")
#
#                 println("\tCell sign = $cellsign")
#                 throw(e)
#             end
#         end
#     end
#     return refclosestpoints, refclosestcellids
# end

function closest_reference_points_on_levelset(
    querypoints,
    refseedpoints,
    spatialseedpoints,
    seedcellids,
    levelset,
    tol,
    boundingradius,
)

    dim, numquerypoints = size(querypoints)
    refclosestpoints = zeros(dim, numquerypoints)
    refclosestcellids = zeros(Int, numquerypoints)
    refgradients = zeros(dim, numquerypoints)

    tree = KDTree(spatialseedpoints)
    seedidx, seeddists = nn(tree, querypoints)

    for (idx, sidx) in enumerate(seedidx)
        xguess = refseedpoints[:, sidx]
        xquery = querypoints[:, idx]
        guesscellid = seedcellids[sidx]

        cellmap = cell_map(background_mesh(levelset), guesscellid)
        load_coefficients!(levelset, guesscellid)
        interpolatingpoly = interpolater(levelset)

        try
            refcp = saye_newton_iterate(
                xguess,
                xquery,
                interpolatingpoly,
                x -> vec(gradient(interpolatingpoly, x)),
                x -> hessian_matrix(interpolatingpoly, x),
                cellmap,
                tol,
                boundingradius,
            )

            refclosestpoints[:, idx] = refcp
            refclosestcellids[idx] = guesscellid
            refgradients[:, idx] = gradient(interpolatingpoly, refcp)
        catch e
            println("Newton iteration failed to converge")
            println("Check levelset function on cell $guesscellid")
            throw(e)
        end
    end
    return refclosestpoints, refclosestcellids, refgradients
end

function distance_to_zero_levelset(
    querypoints,
    refseedpoints,
    spatialseedpoints,
    seedcellids,
    levelset,
    tol,
    boundingradius,
)

    dim, numquerypoints = size(querypoints)
    signeddistance = zeros(numquerypoints)
    jac = jacobian(background_mesh(levelset))

    refclosestpoints, refclosestcellids, refgradients =
        closest_reference_points_on_levelset(
            querypoints,
            refseedpoints,
            spatialseedpoints,
            seedcellids,
            levelset,
            tol,
            boundingradius,
        )

    for i = 1:numquerypoints
        refcp = refclosestpoints[:, i]
        cellmap = cell_map(background_mesh(levelset), refclosestcellids[i])

        spatialcp = cellmap(refcp)
        xquery = querypoints[:, i]


        g = vec(transform_gradient(refgradients[:, i]', jac))
        s = sign(g' * (xquery - spatialcp))

        signeddistance[i] = s * norm(spatialcp - xquery)
    end

    return signeddistance
end
