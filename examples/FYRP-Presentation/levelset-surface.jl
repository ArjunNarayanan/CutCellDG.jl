using Plots;
pyplot();
using LinearAlgebra

function angular_position(points)
    cpoints = points[1] + im * points[2]
    return rad2deg(angle(cpoints))
end

function perturbed_circle(point, meanradius, amplitude, frequency)
    angle = angular_position(point)
    interfaceposition = meanradius + amplitude * sind(angle * frequency)
    distance = interfaceposition - norm(point)
    return distance
end

xrange = -1:1e-1:1
meanradius = 0.5
amplitude = 0.1
frequency = 5

plot(
    xrange,
    xrange,
    (x,y) -> -perturbed_circle([x,y], meanradius, 0., frequency),
    st = :surface,
    #camera = (-30, 30),
)
plot!(xrange,xrange,(x,y)->0.2,st=:surface)
