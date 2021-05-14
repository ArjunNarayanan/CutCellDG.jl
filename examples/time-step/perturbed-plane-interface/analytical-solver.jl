module AnalyticalPlaneSolver

struct PlaneSolver
    L::Any
    W::Any
    R::Any
    lambda1::Any
    mu1::Any
    lambda2::Any
    mu2::Any
    theta0::Any
    C1::Any
    C2::Any
    D::Any
    uI::Any
    function PlaneSolver(L, W, R, lambda1, mu1, lambda2, mu2, theta0)
        C1 = solver_coefficient(lambda1, mu1)
        C2 = solver_coefficient(lambda2, mu2)
        D = 2 * lambda1 * mu1 / (lambda1 + 2mu1)

        uI = interface_displacement(L, R, C1, C2, D, theta0)
        new(L, W, R, lambda1, mu1, lambda2, mu2, theta0, C1, C2, D, uI)
    end
end


function solver_coefficient(lambda, mu)
    C = 4 * mu * (lambda + mu) / (lambda + 2mu)
    return C
end

function interface_displacement(L, R, C1, C2, D, theta0)
    uI = -(C1 + D) / (C2 / R + C1 / (L - R)) * theta0 / 3
    return uI
end

function parent_displacement_field(solver::PlaneSolver,x)
    uI = solver.uI
    R = solver.R
    lambda = solver.lambda2
    mu = solver.mu2

    ux = uI/R*x[1]
    uy = -lambda/(lambda+2mu)*uI/R*x[2]
    return [ux,uy]
end

function product_displacement_field(solver::PlaneSolver,x)
    uI = solver.uI
    L = solver.L
    R = solver.R
    lambda = solver.lambda1
    mu = solver.mu1
    t0 = solver.theta0

    ux = ((L-R)-(x[1]-R))/(L-R)*uI
    uy = (lambda/(lambda+2mu)*uI/(L-R) + (3lambda+2mu)/(lambda+2mu)*t0/3)*x[2]

    return [ux,uy]
end

function displacement_field(solver::PlaneSolver,x)
    R = solver.R
    if x[1] < R
        return parent_displacement_field(solver,x)
    else
        return product_displacement_field(solver,x)
    end
end

end
