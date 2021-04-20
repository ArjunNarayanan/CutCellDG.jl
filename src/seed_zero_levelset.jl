function project_on_zero_levelset(
    xguess,
    func,
    grad,
    tol,
    r;
    maxiter = 20,
    normtol = 1e3eps(),
    perturbation = 1e-2,
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

function tensor_product_points(p1, p2)
    n1 = size(p1)[2]
    n2 = size(p2)[2]
    return vcat(repeat(p1, inner = (1, n2)), repeat(p2, outer = (1, n1)))
end

function reference_seed_points(n)
    @assert n > 0
    xrange = range(-1.0, stop = 1.0, length = n + 2)
    points = tensor_product_points(xrange[2:n+1]', xrange[2:n+1]')
end

function seed_cell_zero_levelset(xguess, func, grad, tol, boundingradius)
    dim, nump = size(xguess)
    projectedpoints = zeros(dim, nump)
    flags = zeros(Bool, nump)
    for idx = 1:nump
        p, f = project_on_zero_levelset(
            xguess[:, idx],
            func,
            grad,
            tol,
            boundingradius,
        )
        projectedpoints[:, idx] = p
        flags[idx] = f
    end

    return projectedpoints, flags
end

function seed_zero_levelset(nump, levelset, cellids, tol, boundingradius)

    dim = dimension(levelset)
    numcells = length(cellids)
    refpoints = reference_seed_points(nump)
    numrefpoints = size(refpoints)[2]
    totalnumpoints = numcells * numrefpoints

    refseedpoints = zeros(dim, totalnumpoints)
    refseedcellids = zeros(Int, totalnumpoints)
    flags = zeros(Bool, totalnumpoints)

    start = 1
    stop = numrefpoints

    for cellid in cellids
        load_coefficients!(levelset, cellid)
        poly = interpolater(levelset)

        xk, f = seed_cell_zero_levelset(
            refpoints,
            poly,
            x -> vec(gradient(poly, x)),
            tol,
            boundingradius,
        )

        refseedpoints[:, start:stop] = xk
        refseedcellids[start:stop] = repeat([cellid], numrefpoints)
        flags[start:stop] = f

        start = stop + 1
        stop = stop + numrefpoints
    end

    validpoints = findall(flags)

    return refseedpoints[:, validpoints], refseedcellids[validpoints]
end

function seed_zero_levelset(
    nump,
    levelset,
    cutmesh;
    tol = 1e4eps(),
    boundingradius = 4.5,
)

    numcells = number_of_cells(cutmesh)
    cellsign = [cell_sign(cutmesh, cellid) for cellid = 1:numcells]
    cellids = findall(cellsign .== 0)
    return seed_zero_levelset(nump, levelset, cellids, tol, boundingradius)
end
