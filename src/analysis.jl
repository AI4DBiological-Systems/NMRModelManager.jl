# plotting helpers.

function setupplotinterval(
    As::Vector{HAM.SHType{T}},
    freq_params::FrequencyParameters{T};
    N_pts::Int = 80000,
    ) where T

    hz2ppmfunc = uu->hz2ppm(uu, freq_params)
    ppm2hzfunc = uu->ppm2hz(uu, freq_params)

    ΩS_ppm = collect( hz2ppmfunc.( SIG.combinevectors(A.Ωs) ./ twopi(T) ) for A in As )
    ΩS_ppm_flat = SIG.combinevectors(ΩS_ppm)
    P_max = maximum(ΩS_ppm_flat) + convert(T, 0.5)
    P_min = minimum(ΩS_ppm_flat) - convert(T, 0.5)
    
    P = LinRange(P_min, P_max, N_pts)
    U = ppm2hzfunc.(P)
    U_rad = U .* twopi(T)

    return P, U, U_rad
end


function batchquery!(out::Vector{Complex{T}}, U_rad::Vector{T}, q) where T <: AbstractFloat
    
    resize!(out, length(U_rad))
    
    for m in eachindex(out)
        out[m] = q(U_rad[m])
    end
    
    return nothing
end