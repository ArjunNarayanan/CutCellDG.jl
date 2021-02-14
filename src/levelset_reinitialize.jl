function seed_zero_levelset_with_interfacequads(interfacequads, mesh)

    ncells = number_of_cells(mesh)
    cellsign = [cell_sign(mesh,cellid) for cellid in 1:ncells]
    cellids = findall(cellsign .== 0)

    return seed_zero_levelset_with_interfacequads(interfacequads,mesh,cellids)
end

function seed_zero_levelset_with_interfacequads(interfacequads, mesh, cellids)

    totalnumqps = sum([length(interfacequads[1,cellid]) for cellid in cellids])
    dim = dimension(mesh)

    refseedpoints = zeros(2, dim, totalnumqps)
    spatialseedpoints = zeros(2, dim, totalnumqps)
    seedcellids = zeros(Int, 2, totalnumqps)

    start = 1
    for cellid in cellids
        cellsign = cell_sign(mesh,cellid)

        if cellsign == 0
            numqps = length(interfacequads[1,cellid])
            stop = start + numqps - 1
            for s in [+1,-1]
                cellmap = cell_map(mesh,s,cellid)
                refpoints = points(interfacequads[s,cellid])
                spatialpoints = cellmap(refpoints)

                row = cell_sign_to_row(s)

                refseedpoints[row,:,start:stop] = refpoints
                spatialseedpoints[row,:,start:stop] = spatialpoints

                seedcellids[row,start:stop] = repeat([cellid],numqps)
            end
            start = stop + 1
        end
    end
    return refseedpoints, spatialseedpoints, seedcellids
end
