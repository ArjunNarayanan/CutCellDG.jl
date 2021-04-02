struct LevelSet
    interpolater
    coefficients
    mesh
    function LevelSet(distancefunc,mesh,basis)
        interpolater = InterpolatingPolynomial(1,basis)
        coeffs = levelset_coefficients(distancefunc,mesh)
        new(interpolater,coeffs,mesh)
    end
end

function coefficients(levelset::LevelSet)
    return levelset.coefficients
end

function coefficients(levelset::LevelSet,cellid)
    mesh = background_mesh(levelset)
    return levelset.coefficients[nodal_connectivity(mesh,cellid)]
end

function background_mesh(levelset::LevelSet)
    return levelset.mesh
end

function interpolater(levelset::LevelSet)
    return levelset.interpolater
end

function load_coefficients!(levelset::LevelSet,cellid)
    update!(interpolater(levelset),coefficients(levelset,cellid))
end

function update_coefficients!(levelset::LevelSet,cellid,coeffs)
    nodeids = nodal_connectivity(levelset.mesh,cellid)
    levelset.coefficients[nodeids] .= coeffs
end

function (levelset::LevelSet)(x)
    levelset.interpolater(x)
end
