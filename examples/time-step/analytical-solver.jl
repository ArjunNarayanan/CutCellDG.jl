module PlaneStrainSolver

function bulk_modulus(l, m)
    return l + 2m / 3
end

function analytical_coefficient_matrix(inradius, outradius, ls, ms, lc, mc)
    a = zeros(3, 3)
    a[1, 1] = inradius
    a[1, 2] = -inradius
    a[1, 3] = -1.0 / inradius
    a[2, 1] = 2 * (lc + mc)
    a[2, 2] = -2 * (ls + ms)
    a[2, 3] = 2ms / inradius^2
    a[3, 2] = 2(ls + ms)
    a[3, 3] = -2ms / outradius^2
    return a
end

function analytical_coefficient_rhs(ls, ms, theta0)
    r = zeros(3)
    Ks = bulk_modulus(ls, ms)
    r[2] = -Ks * theta0
    r[3] = Ks * theta0
    return r
end

struct CylindricalSolver
    inradius::Any
    outradius::Any
    center::Any
    A1c::Any
    A1s::Any
    A2s::Any
    ls::Any
    ms::Any
    lc::Any
    mc::Any
    theta0::Any
    function CylindricalSolver(
        inradius,
        outradius,
        center,
        ls,
        ms,
        lc,
        mc,
        theta0,
    )
        a = analytical_coefficient_matrix(inradius, outradius, ls, ms, lc, mc)
        r = analytical_coefficient_rhs(ls, ms, theta0)
        coeffs = a \ r
        new(
            inradius,
            outradius,
            center,
            coeffs[1],
            coeffs[2],
            coeffs[3],
            ls,
            ms,
            lc,
            mc,
            theta0,
        )
    end
end

function radial_displacement(A::CylindricalSolver, r)
    if r <= A.inradius
        return A.A1c * r
    else
        return A.A1s * r + A.A2s / r
    end
end

function (A::CylindricalSolver)(x)
    relpos = x - A.center
    r = sqrt(relpos' * relpos)
    ur = radial_displacement(A, r)
    if r ≈ 0.0
        [0.0, 0.0]
    else
        costheta = (x[1] - A.center[1]) / r
        sintheta = (x[2] - A.center[2]) / r
        u1 = ur * costheta
        u2 = ur * sintheta
        return [u1, u2]
    end
end

function shell_radial_stress(A::CylindricalSolver, r)
    return (A.ls + 2A.ms) * (A.A1s - A.A2s / r^2) +
           A.ls * (A.A1s + A.A2s / r^2) - (A.ls + 2A.ms / 3) * A.theta0
end

function shell_circumferential_stress(A::CylindricalSolver, r)
    return A.ls * (A.A1s - A.A2s / r^2) +
           (A.ls + 2A.ms) * (A.A1s + A.A2s / r^2) -
           (A.ls + 2A.ms / 3) * A.theta0
end

function shell_out_of_plane_stress(A::CylindricalSolver)
    return 2 * A.ls * A.A1s - (A.ls + 2A.ms / 3) * A.theta0
end

function shell_stress(A::CylindricalSolver, r)

    srr = shell_radial_stress(A, r)
    stt = shell_circumferential_stress(A, r)
    s33 = shell_out_of_plane_stress(A)

    return [srr, stt, s33]
end

function shell_pressure(A::CylindricalSolver, r)
    stress = shell_stress(A, r)
    return pressure(stress)
end

function shell_radial_strain(A::CylindricalSolver, r)
    return A.A1s - A.A2s / r^2 - A.theta0 / 3
end

function shell_circumferential_strain(A::CylindricalSolver, r)
    return A.A1s + A.A2s / r^2 - A.theta0 / 3
end

function shell_out_of_plane_strain(A::CylindricalSolver)
    return -A.theta0 / 3
end

function shell_strain(A::CylindricalSolver, r)
    err = shell_radial_strain(A, r)
    ett = shell_circumferential_strain(A, r)
    ezz = shell_out_of_plane_strain(A)
    return [err, ett, ezz]
end

function shell_strain_energy(A::CylindricalSolver, r, V0)
    stress = shell_stress(A, r)
    strain = shell_strain(A, r)

    return V0 * 0.5 * (sum(stress .* strain))
end

function shell_dilatation(A::CylindricalSolver)
    return 2 * A.A1s - A.theta0
end

function shell_compression_work(A::CylindricalSolver, r, V0)
    ekk = shell_dilatation(A)
    srr = shell_radial_stress(A, r)
    return V0 * (1 + ekk) * srr
end

function shell_potential(A::CylindricalSolver, r, V0)
    strainenergy = shell_strain_energy(A, r, V0)
    compwork = shell_compression_work(A, r, V0)
    return strainenergy - compwork
end

###########################################################################
function core_in_plane_stress(A::CylindricalSolver)
    return (A.lc + 2A.mc) * A.A1c + A.lc * A.A1c
end

function core_in_plane_strain(A::CylindricalSolver)
    return A.A1c
end

function core_strain(A::CylindricalSolver)
    err = core_in_plane_strain(A)
    return [err, err, 0.0]
end

function core_out_of_plane_stress(A::CylindricalSolver)
    return 2 * A.lc * A.A1c
end

function core_stress(A::CylindricalSolver)
    s11 = core_in_plane_stress(A)
    s33 = core_out_of_plane_stress(A)
    return [s11, s11, s33]
end

function core_pressure(A::CylindricalSolver)
    stress = core_stress(A)
    return pressure(stress)
end

function core_strain_energy(A::CylindricalSolver, V0)
    srr = core_in_plane_stress(A)
    err = core_in_plane_strain(A)
    stt = srr
    ett = err
    return V0 * 0.5 * (srr * err + stt * ett)
end

function core_dilatation(A::CylindricalSolver)
    return 2 * A.A1c
end

function core_compression_work(A::CylindricalSolver, V0)
    srr = core_in_plane_stress(A)
    ekk = core_dilatation(A)
    return V0 * (1 + ekk) * srr
end

function core_potential(A::CylindricalSolver, V0)
    strainenergy = core_strain_energy(A, V0s)
    compwork = core_compression_work(A, V0)
    return strainenergy - compwork
end

###########################################################################

function interface_potential_difference(solver,V01,V02)
    cse = core_strain_energy(solver,V02)
    ccw = core_compression_work(solver,V02)

    sse = shell_strain_energy(solver,solver.inradius,V01)
    scw = shell_compression_work(solver,solver.inradius,V01)

    return (sse - scw) - (cse - ccw)
end

function pressure(stress)
    return -1.0 / 3.0 * sum(stress)
end

end
