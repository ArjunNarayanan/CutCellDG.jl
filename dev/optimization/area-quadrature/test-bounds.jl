function split_box(box, numsplits)
    return mince(box, numsplits)
end

function min_diam(box)
    return minimum(diam.(box))
end

function interval_arithmetic_sign_search(
    func,
    initialbox,
    tol,
    perturbation,
    numsplits,
)

    rtol = tol * diam(initialbox)
    numiter = 0

    foundpos = foundneg = breachedtol = false
    queue = [initialbox]

    while !isempty(queue)
        if (foundpos && foundneg)
            break
        else
            box = popfirst!(queue)
            if min_diam(box) < rtol
                breachedtol = true
                break
            end
            funcrange = func(box)
            if inf(funcrange) > -perturbation
                foundpos = true
            elseif sup(funcrange) < perturbation
                foundneg = true
            else
                newboxes = split_box(box, numsplits)
                push!(queue, newboxes...)
            end
        end
        numiter += 1
    end

    returnval = 2
    if breachedtol
        returnval = 2
    elseif foundpos && foundneg
        returnval = 0
    elseif foundpos && !foundneg
        returnval = +1
    elseif !foundpos && foundneg
        returnval = -1
    else
        error("Unexpected scenario")
    end

    return returnval, numiter
end
