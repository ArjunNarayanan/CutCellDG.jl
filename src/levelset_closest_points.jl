function spatial_hessian(poly, reference_x, invjac)
    h = hessian(poly, reference_x)
    H11 = h[1] * invjac[1]^2
    H12 = h[2] * invjac[1] * invjac[2]
    H22 = h[3] * invjac[2]^2
    return [
        H11 H12
        H12 H22
    ]
end

function spatial_gradient(poly, reference_x, invjac)
    return vec(gradient(poly, reference_x)) .* invjac
end

function saye_newton_gradient(xk, lk, xq, pk, ∇pk)
    return vcat(xk - xq + lk * ∇pk, pk)
end

function saye_newton_hessian(lk, ∇pk, ∇2pk)
    H = [
        I+lk*∇2pk ∇pk
        ∇pk' 0.0
    ]
    return H
end

function step_saye_newton_iterate(xk, lk, ∇f, ∇2f, steplimiter)
    dim = length(xk)

    δ = ∇2f \ ∇f
    normδx = norm(δ[1:dim])
    if normδx > steplimiter
        δ *= steplimiter / normδx
    end
    xnext = xk - δ[1:dim]
    lnext = lk - δ[end]
    return xnext, lnext
end

function step_chopp_iterate(xk, xq, pk, ∇pk, steplimiter)
    norm2∇pk = dot(∇pk, ∇pk)

    δ1 = -pk / norm2∇pk * ∇pk
    lnext = (xq - xk)' * ∇pk / norm2∇pk

    δ2 = xq - xk - lnext * ∇pk
    normδ2 = norm(δ2)
    if normδ2 > steplimiter
        δ2 *= steplimiter / normδ2
    end
    xnext = xk + δ1 + δ2

    return xnext, lnext
end

function step_saye_iterate(
    xk,
    Xk,
    lk,
    xq,
    func,
    grad,
    hess,
    condtol,
    steplimiter,
)

    dim = length(xk)
    pk = func(Xk)
    ∇pk = grad(Xk)
    ∇2pk = hess(Xk)

    gf = saye_newton_gradient(xk, lk, xq, pk, ∇pk)
    hf = saye_newton_hessian(lk, ∇pk, ∇2pk)

    if inv(cond(hf)) > condtol
        return step_saye_newton_iterate(xk, lk, gf, hf, steplimiter)
    else
        return step_chopp_iterate(xk, xq, pk, ∇pk, steplimiter / 5)
    end
end

function spatial_closest_point(
    xquery,
    xguess,
    func,
    cellmap,
    tol,
    boundingradius,
    maxiter,
    condtol,
)

    invjac = inverse_jacobian(cellmap)
    grad(X) = spatial_gradient(func, X, invjac)
    hess(X) = spatial_hessian(func, X, invjac)

    x0 = copy(xguess)
    X0 = inverse(cellmap, x0)
    g0 = grad(X0)
    l0 = (xquery - x0)' * g0 / (g0' * g0)

    x1 = copy(x0)
    steplimiter = boundingradius / 2
    for iter = 1:maxiter
        X0 = inverse(cellmap, x0)
        x1, l1 = step_saye_iterate(
            x0,
            X0,
            l0,
            xquery,
            func,
            grad,
            hess,
            condtol,
            steplimiter,
        )

        if norm(x1 - xguess) > boundingradius
            return x0, false
        elseif norm(x1 - x0) < tol
            return x1, true
        else
            x0 = x1
            l0 = l1
        end
    end
    return x1, false
end

function closest_points_on_zero_levelset(
    querypoints,
    seedpoints,
    seedcellids,
    levelset,
    tol,
    boundingradius;
    condtol = 1e4eps(),
    maxiter = 20,
)

    mesh = background_mesh(levelset)
    dim, numquerypoints = size(querypoints)
    closestpoints = zeros(dim, numquerypoints)
    closestcellids = zeros(Int, numquerypoints)
    flags = zeros(Bool, numquerypoints)

    tree = KDTree(seedpoints)
    seedidx, seeddists = nn(tree, querypoints)

    for (idx, sidx) in enumerate(seedidx)
        xguess = seedpoints[:, sidx]
        xquery = querypoints[:, idx]
        guesscellid = seedcellids[sidx]

        cellmap = cell_map(mesh, guesscellid)
        load_coefficients!(levelset, guesscellid)
        func = interpolater(levelset)

        xcp, f = spatial_closest_point(
            xquery,
            xguess,
            func,
            cellmap,
            tol,
            boundingradius,
            maxiter,
            condtol,
        )

        closestpoints[:, idx] = xcp
        closestcellids[idx] = guesscellid
        flags[idx] = f
    end
    return closestpoints, closestcellids, flags
end

function spatial_gradient_at_reference_points(levelset, refpoints, refcellids)
    invjac = inverse_jacobian(background_mesh(levelset))
    dim, numpts = size(refpoints)
    spatialgrads = zeros(dim, numpts)
    for (idx, cellid) in enumerate(refcellids)
        load_coefficients!(levelset, cellid)
        poly = interpolater(levelset)
        point = refpoints[:, idx]
        spatialgrads[:,idx] = spatial_gradient(poly, point, invjac)
    end
    return spatialgrads
end

function map_to_reference(spatialpoints, cellids, mesh)
    referencepoints = similar(spatialpoints)
    for (idx, cellid) in enumerate(cellids)
        cellmap = cell_map(mesh, cellid)
        referencepoints[:, idx] = inverse(cellmap, spatialpoints[:, idx])
    end
    return referencepoints
end

function signed_distance(
    querypoints,
    spatialclosestpoints,
    refclosestpoints,
    cellids,
    levelset,
)

    spatialgrads = spatial_gradient_at_reference_points(
        levelset,
        refclosestpoints,
        cellids,
    )

    difference = querypoints - spatialclosestpoints
    levelsetsign = sign.(vec(sum(spatialgrads .* difference, dims = 1)))

    signeddistance = levelsetsign .* vec(mapslices(norm, difference, dims = 1))

    return signeddistance
end

function distance_to_zero_levelset(
    querypoints,
    seedpoints,
    seedcellids,
    levelset,
    tol,
    boundingradius,
)
    closestpoints, closestcellids = closest_points_on_zero_levelset(
        querypoints,
        seedpoints,
        seedcellids,
        levelset,
        tol,
        boundingradius,
    )
    mesh = background_mesh(levelset)
    refclosestpoints = map_to_reference(closestpoints, closestcellids, mesh)
    signeddistance = signed_distance(
        querypoints,
        closestpoints,
        refclosestpoints,
        closestcellids,
        levelset,
    )

    return signeddistance
end
