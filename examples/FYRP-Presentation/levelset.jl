using Plots
using LinearAlgebra

function angular_position(points)
    cpoints = points[1] + im * points[2]
    return rad2deg(angle(cpoints))
end

function perturbed_circle(point, meanradius, amplitude, frequency)
    angle = angular_position(point)
    interfaceposition = meanradius  + amplitude * sind(angle * frequency)
    distance = interfaceposition - norm(point)
    return distance
end

xrange = -1:1e-2:1
meanradius = 0.5
amplitude = 0.1
frequency = 5
contour(
    xrange,
    xrange,
    (x, y) ->
        perturbed_circle([x, y], meanradius, amplitude, frequency),
        levels=[0.0],
        aspect_ratio=:equal
)
