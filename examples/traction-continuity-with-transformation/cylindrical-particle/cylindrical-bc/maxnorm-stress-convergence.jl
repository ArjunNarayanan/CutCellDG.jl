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

function interface_maxnorm_stress_error(polyorder, nelmts, penaltyfactor)
    width = 1.0
    K1, K2 = 247.0, 192.0    # Pa
    mu1, mu2 = 126.0, 87.0   # Pa
    lambda1 = lame_lambda(K1, mu1)
    lambda2 = lame_lambda(K2, mu2)
    stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)

    theta0 = -0.067
    transfstress =
        CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)

    meshwidth = [width, width]
    numqp = required_quadrature_order(polyorder) + 2

    dx = width / nelmts
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

    productradialtraction =
        CutCellDG.traction_force_at_points(productstress, normals)
    parentradialtraction =
        CutCellDG.traction_force_at_points(parentstress, normals)

    productradialstress = vec(sum(productradialtraction .* normals, dims = 1))
    parentradialstress = vec(sum(parentradialtraction .* normals, dims = 1))

    exactproductradialstress =
        shell_radial_stress(analyticalsolution, interfaceradius)
    exactparentradialstress = core_in_plane_stress(analyticalsolution)

    productradialstressmaxnormerror =
        maximum(abs.(productradialstress .- exactproductradialstress)) /
        abs(exactproductradialstress)
    parentradialstressmaxnormerror =
        maximum(abs.(parentradialstress .- exactparentradialstress)) /
        abs(exactparentradialstress)

    productcircumferentialtraction =
        CutCellDG.traction_force_at_points(productstress, tangents)
    productcircumferentialstress =
        vec(sum(productcircumferentialtraction .* tangents, dims = 1))

    exactproductcircumferentialstress =
        shell_circumferential_stress(analyticalsolution, interfaceradius)

    productcircumferentialstressmaxnormerror =
        maximum(
            abs.(
                productcircumferentialstress .-
                exactproductcircumferentialstress,
            ),
        ) / abs(exactproductcircumferentialstress)

    return [
        parentradialstressmaxnormerror,
        productradialstressmaxnormerror,
        productcircumferentialstressmaxnormerror,
    ]
end

polyorder = 3
penaltyfactor = 1e2
powers = [3, 4, 5]
nelmts = [2^p+1 for p in powers]
dx = 1.0 ./ nelmts

stresserror = [
    interface_maxnorm_stress_error(polyorder, ne, penaltyfactor) for
    ne in nelmts
]

stresserror = vcat([s' for s in stresserror]...)

using CSV, DataFrames

grid = [string(ne)*" x "*string(ne) for ne in nelmts]

df = DataFrame(
    "Grid" => grid,
    "Parent Radial Stress Error" => stresserror[:, 1],
    "Product Radial Stress Error" => stresserror[:, 2],
    "Product Circumferential Stress Error" => stresserror[:,3]
)

folderpath = "examples/traction-continuity-with-transformation/cylindrical-particle/cylindrical-bc/"
filename = "polyorder-"*string(polyorder)*".csv"
CSV.write(folderpath*filename,df)
