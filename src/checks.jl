
function checkNresonancegroups(As::Vector{HAM.SHType{T}}) where T

    valid_flag = true
    for n in eachindex(As)
        @assert length(As[n].Δc_bar) == length(As[n].Δc)
        for i in eachindex(As[n].Δc)
            N_nuclei = length(As[n].Δc[i][begin])
            N_resonance_groups = length(As[n].Δc_bar[i])
            if N_resonance_groups < N_nuclei
                println("Warning, spin system (n, i) = ($n, $i) has $N_resonance_groups of resonance groups, but $N_nuclei nuclei. You might want to re-simulate the model using a smaller γ_base parameter, unless you know the compounds have fewwer degress of freedom than the number of magnetically equivalent nuclei.")
                valid_flag = false
            end
        end
    end

    return valid_flag
end

