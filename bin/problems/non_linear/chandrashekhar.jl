function p23_f!(out, x, p = nothing)
    c = 0.5
    n = size(out, 1)
    out[1:n] = x[1:n]
    μ = zeros(n)
    for i in 1:n
        μ[i] = (i - 1 / 2) / n
    end
    for i in 1:n
        s = 0.0
        for j in 1:n
            s = s + (μ[i] * x[j]) / (μ[i] + μ[j])
        end
        term = 1.0 - c * s / (2 * n)
        out[i] -= 1.0 / term
    end
    nothing
end
