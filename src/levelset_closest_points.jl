function hessian_matrix(poly, x)
    h = hessian(poly, x)
    return [
        h[1] h[2]
        h[2] h[3]
    ]
end
