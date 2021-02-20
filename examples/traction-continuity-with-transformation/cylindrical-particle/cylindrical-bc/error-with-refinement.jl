using LinearAlgebra
using PyPlot
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("../../../../test/useful_routines.jl")
include("transformation-elasticity-solver.jl")

function angular_position(points)
    cpoints = points[1, :] + im * points[2, :]
    return rad2deg.(angle.(cpoints))
end


function stress_error_vs_angular_position(polyorder, nelmts, penaltyfactor)
    K1, K2 = 247.0, 192.0    # Pa
    mu1, mu2 = 126.0, 87.0   # Pa
    lambda1 = lame_lambda(K1, mu1)
    lambda2 = lame_lambda(K2, mu2)
    stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)

    theta0 = -0.067
    transfstress =
        CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)

    meshwidth = [1.0, 1.0]
    numqp = required_quadrature_order(polyorder) + 2

    dx = minimum(meshwidth) / nelmts
    penalty = penaltyfactor / dx * 0.5 * (lambda1 + mu1 + lambda2 + mu2)

    basis = TensorProductBasis(2, polyorder)
    interfacecenter = [0.0, 0.0]
    interfaceradius = 0.5
    outerradius = 2.0
    analyticalsolution = AnalyticalSolution(
        interfaceradius,
        outerradius,
        interfacecenter,
        lambda1,
        mu1,
        lambda2,
        mu2,
        theta0,
    )

    mesh, cellquads, facequads, interfacequads = construct_mesh_and_quadratures(
        meshwidth,
        nelmts,
        basis,
        interfacecenter,
        interfaceradius,
        numqp,
        tinyratio = 0.2,
    )

    nodaldisplacement = nodal_displacement(
        mesh,
        basis,
        cellquads,
        facequads,
        interfacequads,
        stiffness,
        theta0,
        analyticalsolution,
        penalty,
    )

    refseedpoints, spatialseedpoints, seedcellids =
        CutCellDG.seed_zero_levelset_with_interfacequads(interfacequads, mesh)

    normals = CutCellDG.collect_interface_normals(interfacequads, mesh)
    tangents = CutCellDG.rotate_90(normals)

    spatialpoints = spatialseedpoints[1, :, :]
    referencepoints = refseedpoints
    referencecellids = seedcellids

    relspatialpoints = spatialpoints .- interfacecenter
    angularposition = angular_position(relspatialpoints)
    sortidx = sortperm(angularposition)
    angularposition = angularposition[sortidx]

    referencepoints = referencepoints[:, :, sortidx]
    referencecellids = referencecellids[:, sortidx]
    spatialpoints = spatialpoints[:, sortidx]
    normals = normals[:, sortidx]
    tangents = tangents[:, sortidx]

    productstress = product_stress_at_reference_points(
        nodaldisplacement,
        basis,
        stiffness,
        transfstress,
        theta0,
        referencepoints,
        referencecellids,
        mesh,
    )
    parentstress = parent_stress_at_reference_points(
        nodaldisplacement,
        basis,
        stiffness,
        referencepoints,
        referencecellids,
        mesh,
    )
    exactparentstress =
        mapslices(x -> core_stress(analyticalsolution), spatialpoints, dims = 1)
    exactproductstress = mapslices(
        x -> shell_stress(analyticalsolution, x),
        spatialpoints,
        dims = 1,
    )


    productradialtraction =
        CutCellDG.traction_force_at_points(productstress, normals)
    parentradialtraction =
        CutCellDG.traction_force_at_points(parentstress, normals)

    productradialstress = vec(sum(productradialtraction .* normals, dims = 1))
    parentradialstress = vec(sum(parentradialtraction .* normals, dims = 1))

    exactproductradialstress =
        shell_radial_stress(analyticalsolution, interfaceradius)
    exactparentradialstress = core_in_plane_stress(analyticalsolution)

    productradialstresserror =
        abs.(productradialstress .- exactproductradialstress) ./
        abs(exactproductradialstress)
    parentradialstresserror =
        abs.(parentradialstress .- exactparentradialstress) ./
        abs(exactparentradialstress)

    productcircumferentialtraction =
        CutCellDG.traction_force_at_points(productstress, tangents)
    productcircumferentialstress =
        vec(sum(productcircumferentialtraction .* tangents, dims = 1))

    exactproductcircumferentialstress =
        shell_circumferential_stress(analyticalsolution, interfaceradius)
    productcircumferentialstresserror =
        abs.(
            productcircumferentialstress .- exactproductcircumferentialstress,
        ) ./ abs(exactproductcircumferentialstress)

    return angularposition,
    parentradialstresserror,
    productradialstresserror,
    productcircumferentialstresserror

end

polyorder = 3
results9 = stress_error_vs_angular_position(polyorder, 9, 1e2)
results17 = stress_error_vs_angular_position(polyorder, 17, 1e2)
results33 = stress_error_vs_angular_position(polyorder, 33, 1e2)

folderpath = "examples/traction-continuity-with-transformation/cylindrical-particle/cylindrical-bc/"
filelabel = "polyorder-"*string(polyorder)*"-"

fig, ax = PyPlot.subplots()
ax.plot(results9[1], results9[2], label = "9 x 9")
ax.plot(results17[1], results17[2], label = "17 x 17")
ax.plot(results33[1], results33[2], label = "33 x 33")
ax.legend()
ax.set_ylim(0,0.005)
ax.grid()
ax.set_xlabel("Angular position (deg)")
ax.set_ylabel("Normalized error in parent radial stress")
fig.tight_layout()
fig
fig.savefig(folderpath*filelabel*"parent-radial-stress-error.png")

fig, ax = PyPlot.subplots()
ax.plot(results9[1], results9[3], label = "9 x 9")
ax.plot(results17[1], results17[3], label = "17 x 17")
ax.plot(results33[1], results33[3], label = "33 x 33")
ax.legend()
ax.grid()
ax.set_ylim(0,0.005)
ax.set_xlabel("Angular position (deg)")
ax.set_ylabel("Normalized error in product radial stress")
fig.tight_layout()
fig
fig.savefig(folderpath*filelabel*"product-radial-stress-error.png")

fig, ax = PyPlot.subplots()
ax.plot(results9[1], results9[4], label = "9 x 9")
ax.plot(results17[1], results17[4], label = "17 x 17")
ax.plot(results33[1], results33[4], label = "33 x 33")
ax.legend()
ax.grid()
ax.set_ylim(0,0.005)
ax.set_xlabel("Angular position (deg)")
ax.set_ylabel("Normalized error in product circumferential stress")
fig.tight_layout()
fig
fig.savefig(folderpath*filelabel*"product-circumferential-stress-error.png")
