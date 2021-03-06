using PyPlot
using Statistics
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCellDG
include("../../../../test/useful_routines.jl")
include("../transformation-elasticity-solver.jl")
include("analytical-solver.jl")
TES = TransformationElasticitySolver
APS = AnalyticalPlaneSolver


function interface_potential(
    querypoints,
    distancefunction,
    nelmts,
    polyorder,
    penaltyfactor,
)

    meshwidth = [1.0, 1.0]
    numqp = required_quadrature_order(polyorder) + 2

    K1, K2 = 247.0, 247.0    # GPa
    mu1, mu2 = 126.0, 126.0   # GPa

    lambda1 = lame_lambda(K1, mu1)
    lambda2 = lame_lambda(K2, mu2)
    stiffness = CutCellDG.HookeStiffness(lambda1, mu1, lambda2, mu2)
    rho1 = 3.93e3           # Kg/m^3
    rho2 = 3.68e3           # Kg/m^3
    V01 = 1.0 / rho1
    V02 = 1.0 / rho2
    ΔG0Jmol = -14351.0
    molarmass = 0.147
    ΔG0 = ΔG0Jmol / molarmass
    theta0 = -0.067
    transfstress =
        CutCellDG.plane_strain_transformation_stress(lambda1, mu1, theta0)

    solverbasis = TensorProductBasis(2, polyorder)
    basispoints = interpolation_points(solverbasis)
    dim,numpts = size(basispoints)
    cgmesh = CutCellDG.CGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], numpts)
    dgmesh = CutCellDG.DGMesh([0.0, 0.0], meshwidth, [nelmts, nelmts], basispoints)
    levelset = CutCellDG.LevelSet(distancefunction, cgmesh, solverbasis)

    elementsize = CutCellDG.element_size(cgmesh)
    minelmtsize = minimum(elementsize)
    maxelmtsize = maximum(elementsize)
    penalty =
        penaltyfactor / minelmtsize * 0.5 * (lambda1 + mu1 + lambda2 + mu2)
    tol = minelmtsize^(polyorder + 1)
    boundingradius = 1.5 * maxelmtsize

    cutmesh = CutCellDG.CutMesh(dgmesh, levelset)
    refseedpoints, seedcellids =
        CutCellDG.seed_zero_levelset(2, levelset, cutmesh)
    spatialseedpoints =
        CutCellDG.map_to_spatial(refseedpoints, seedcellids, cutmesh)

    # pd = TES.potential_difference_at_query_points(
    #     querypoints,
    #     cutmesh,
    #     solverbasis,
    #     levelset,
    #     stiffness,
    #     theta0,
    #     numqp,
    #     penalty,
    #     spatialseedpoints,
    #     seedcellids,
    #     V01,
    #     V02,
    #     ΔG0,
    #     tol,
    #     boundingradius,
    # )
    pd = TES.overpotential_difference_at_query_points(
        querypoints,
        cutmesh,
        solverbasis,
        levelset,
        stiffness,
        theta0,
        numqp,
        penalty,
        spatialseedpoints,
        seedcellids,
        V01,
        V02,
        tol,
        boundingradius,
    )


    return pd
end

function perturbation(x, frequency, amplitude)
    return amplitude * sin.(2 * pi * frequency * x)
end

function plot_zero_levelset(coeffs, nodalcoordinates)
    fig, ax = PyPlot.subplots()
    ax.tricontour(nodalcoordinates[1, :], nodalcoordinates[2, :], coeffs, [0.0])
    ax.grid()
    ax.set_aspect("equal")
    return fig
end

distancefunction(x, initialposition, frequency, amplitude) =
    plane_distance_function(x, [1.0, 0.0], [initialposition, 0.0]) +
    perturbation(x[2, :], frequency, amplitude)

function plot_interface_and_potential(
    ycoords,
    potential,
    initialposition,
    frequency,
    amplitude,
    interfacescale,
    pdscale;
    filepath = "",
)
    interfaceposition =
        initialposition .+ perturbation.(ycoords, frequency, amplitude)
    planeinterfacepd = mean(potential)

    interfaceylim = 5 * amplitude
    pdylim = pdscale

    fig, ax = PyPlot.subplots(2, 1, sharex = true)
    ax[1].plot(ycoords, interfaceposition, color = "black")
    ax[1].set_ylim(
        initialposition - interfaceylim,
        initialposition + interfaceylim,
    )
    ax[1].grid()
    ax[1].set_ylabel("Interface position")

    ax[2].plot(ycoords, potential, color = "black")
    ax[2].set_ylim(planeinterfacepd - pdylim, planeinterfacepd + pdylim)
    ax[2].grid()
    ax[2].set_ylabel("Potential difference")
    ax[2].set_xlabel("y")
    fig.tight_layout()

    if length(filepath) > 0
        fig.savefig(filepath)
        return fig
    else
        return fig
    end
end


initialposition = 0.5
frequency = 2.5
# amplitude = 1e-2
amplitude = 0.0

numquerypoints = 1000
querypoints = vcat(
    repeat([initialposition], numquerypoints)',
    range(0, 1, length = numquerypoints)',
)
ycoords = querypoints[2, :]

# polyorder = 2
# nelmts = 33
# penaltyfactor = 1e1
# pd = interface_potential(
#     querypoints,
#     x -> distancefunction(x, initialposition, frequency, amplitude),
#     nelmts,
#     polyorder,
#     penaltyfactor,
# )
#
#
# pdnoise = (maximum(pd) - minimum(pd)) / abs(mean(pd))
# pddiff = (maximum(pd) - minimum(pd)) / 2
# println("PD Noise = $pdnoise")
#
# interfacescale = 5
# pdscale = 1e-8
# plot_interface_and_potential(
#     ycoords,
#     pd,
#     initialposition,
#     frequency,
#     amplitude,
#     interfacescale,
#     pdscale,
# )



# foldername =
#     "examples\\time-step\\perturbed-plane-interface\\" * string(amplitude)
# if !isdir(foldername)
#     mkdir(foldername)
# end
# filename =
#     "\\polyorder-" *
#     string(polyorder) *
#     "-nelmts-" *
#     string(nelmts) *
#     "-penalty-" *
#     string(penaltyfactor) *
#     "-perturbation-" *
#     string(frequency) *
#     ".png"
# filepath = foldername * filename
# plot_interface_and_potential(
#     ycoords,
#     pd,
#     initialposition,
#     frequency,
#     amplitude,
#     interfacescale,
#     pdscale,
#     filepath = filepath,
# )
