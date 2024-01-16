
struct VarInfo
    label::Symbol
    compound::Int
    spin_sys::Int # within the compound.
    nuclei::Int # within the spin system.
end

###### mapping-related, for data generate or optimization variable preparation/modification.

function getparamsmapping(MSS::SIG.MixtureSpinSys)::Tuple{Vector{Vector{UnitRange{Int}}},Vector{Vector{UnitRange{Int}}},Vector{Vector{UnitRange{Int}}},SIG.ParamsMapping}
    
    shifts, phases, T2s = MSS.shifts, MSS.phases, MSS.T2s
    params_mapping = SIG.ParamsMapping(shifts, phases, T2s)
    pm = params_mapping

    N_compounds = getNcompounds(MSS)
    shift_group_ranges = Vector{Vector{UnitRange{Int}}}(undef, N_compounds)
    phase_group_ranges = Vector{Vector{UnitRange{Int}}}(undef, N_compounds)
    T2_group_ranges = Vector{Vector{UnitRange{Int}}}(undef, N_compounds)

    # shift.
    for n in eachindex(pm.shift.st)
        
        N_sys_n = getNsys(MSS, n)
        shift_group_ranges[n] = Vector{UnitRange{Int}}(undef, N_sys_n)
        
        for i in eachindex(pm.shift.st[n])
            st = pm.shift.st[n][i]
            fin = pm.shift.fin[n][i]
            shift_group_ranges[n][i] = st:fin
        end
    end

    # phase.
    for n in eachindex(pm.phase.st)
        
        N_sys_n = getNsys(MSS, n)
        phase_group_ranges[n] = Vector{UnitRange{Int}}(undef, N_sys_n)
        
        for i in eachindex(pm.phase.st[n])
            st = pm.phase.st[n][i]
            fin = pm.phase.fin[n][i]
            phase_group_ranges[n][i] = st:fin
        end
    end

    # T2.
    for n in eachindex(pm.T2.st)
        
        N_sys_n = getNsys(MSS, n)
        T2_group_ranges[n] = Vector{UnitRange{Int}}(undef, N_sys_n)
        
        for i in eachindex(pm.T2.st[n])
            st = pm.T2.st[n][i]
            fin = pm.T2.fin[n][i]
            T2_group_ranges[n][i] = st:fin
        end
    end

    return shift_group_ranges, phase_group_ranges, T2_group_ranges, params_mapping
end

function getNcompounds(MSS::SIG.MixtureSpinSys)::Int
    return length(MSS.shifts)
end

function getNsys(MSS::SIG.MixtureSpinSys)::Int
    return sum( getNsys(MSS, n) for n in eachindex(MSS.shifts) )
end

function getNsys(MSS::SIG.MixtureSpinSys, n::Int)::Int
    @assert 1 <= n <= length(MSS.shifts)

    return length( MSS.shifts[n].var )
end

function collectflatten(shift_group_ranges::Vector{Vector{UnitRange{Int}}})::Vector{UnitRange{Int}}
    return collect( Iterators.flatten(shift_group_ranges) )
end

function getNsys(shift_group_ranges::Vector{Vector{UnitRange{Int}}})::Int
    return sum( 1 for _ in Iterators.flatten(shift_group_ranges))
end

function getNentries(model::ModelContainer)::Int
    return length(model.molecule_entries)
end

function getparamsLUT(model::ModelContainer)::Vector{VarInfo}
    return getparamsLUT(model.MSS)
end

function getparamsLUT(MSS::SIG.MixtureSpinSys)::Vector{VarInfo}

    shift_group_ranges, phase_group_ranges, T2_group_ranges, _ = getparamsmapping(MSS)

    N_vars = T2_group_ranges[end][end][end]
    var_infos = Vector{VarInfo}(undef, N_vars)

    k = 0
    for n in eachindex(shift_group_ranges)
        for i in eachindex(shift_group_ranges[n])

            j = 0
            for _ in shift_group_ranges[n][i]
                k += 1
                j += 1
                var_infos[k] = VarInfo( :shift, n, i, j )
            end
        end
    end

    for n in eachindex(phase_group_ranges)
        for i in eachindex(phase_group_ranges[n])
            
            j = 0
            for _ in phase_group_ranges[n][i]
                k += 1
                j += 1
                var_infos[k] = VarInfo( :phase, n, i, j )
            end
        end
    end

    for n in eachindex(T2_group_ranges)
        for i in eachindex(T2_group_ranges[n])

            j = 0
            for _ in T2_group_ranges[n][i]
                k += 1
                j += 1
                var_infos[k] = VarInfo( :T2, n, i, j )
            end
        end
    end

    return var_infos
end


function getdefaultparameter(::Type{T}, LUT::Vector{VarInfo}) where T

    N_vars = length(LUT)
    p = zeros(T, N_vars)
    resetparameter!(p, LUT, :T2, one(T))
    resetparameter!(p, LUT, :shift, zero(T))
    resetparameter!(p, LUT, :phase, zero(T))

    return p
end


# reset the values of all parameters with the `target` tag in `LUT`.
function resetparameter!(p::Vector{T}, LUT::Vector{VarInfo}, target::Symbol, target_val::T) where T

    @assert length(p) == length(LUT)

    for i in eachindex(p)
        if LUT[i].label == target
            p[i] = target_val
        end
    end

    return nothing
end

# rewrite a specific type of variable bounds.
function replacebounds!(
    lbs::Vector{T}, # mutates
    target_label::Symbol,
    new_value::T,
    params_LUT::Vector{VarInfo},
    ) where T <: AbstractFloat

    inds = findall(xx->xx.label==target_label, params_LUT)
    for i in inds
        lbs[i] = new_value
    end

    return nothing
end

# rewrite a specific type of variable bounds.
function narrowshiftbounds!(
    lbs::Vector{T}, # mutates for entries that are smaller than new_lb.
    ubs::Vector{T}, # mutates for entries that are larger than new_ub.
    new_lb::T,
    new_ub::T,
    params_LUT::Vector{VarInfo},
    ) where T <: AbstractFloat

    @assert length(lbs) == length(ubs)
    
    inds = findall(xx->xx.label==:shift, params_LUT)
    for i in inds
        if lbs[i] < new_lb
            lbs[i] = new_lb
        end

        if ubs[i] > new_ub
            ubs[i] = new_ub
        end
    end

    return nothing
end

##################### for plotting and isolating shift variables in an interval.

function identifyshifts(
    band_inds::Vector{Vector{Int}},
    U_y,
    cs::Vector{T},
    freq_params::FrequencyParameters,
    ) where T
    
    hz2ppmfunc = uu->hz2ppm(uu, freq_params)

    shift_inds = Vector{Vector{Int}}(undef, length(band_inds))
    for m in eachindex(band_inds)
        
        U = U_y[band_inds[m]]
        P = hz2ppmfunc.(U)
        min_P = minimum(P)
        max_P = maximum(P)

        shift_inds[m] = findall(xx->(min_P <= xx <= max_P), cs)
    end

    return shift_inds
end

# returns a flatten chemical shift array of all compounds. Takes into account of magnetic equivalence.
function getcs(Phys::Vector{HAM.PhysicalParamsType{T}})::Vector{T} where T
    
    cs_2x_nested = collect( HAM.readbasechemshifts(Phys[n]) for n in eachindex(Phys) )
    cs = collect( Iterators.flatten( Iterators.flatten(cs_2x_nested)) )

    return cs
end

function getcsinterval(Phys::Vector{HAM.PhysicalParamsType{T}})::Tuple{T,T} where T

    cs = getcs(Phys)

    return minimum(cs), maximum(cs)
end


################### for creating subset_vars::SIG.SubsetVars

# assuming CoherenceShift, CoherencePhase, SharedT2 parameter mapping.
function findvars(
    LUT::Vector{VarInfo},
    shift_indices::Vector{Int},
    )::SIG.SubsetVars

    #shift_indices = cs_inds[m]
    active_systems_sp = collect( (LUT[i].compound, LUT[i].spin_sys) for i in shift_indices )

    offset = findfirst(xx->xx.label == :phase, LUT) -1
    phase_indices = shift_indices .+ offset
    
    offset = findfirst(xx->xx.label == :T2, LUT) -1
    T2_indices = Vector{Int}(undef, 0)
    for k in Iterators.drop(eachindex(LUT), offset)
    
        r = findfirst(xx->xx == (LUT[k].compound, LUT[k].spin_sys), active_systems_sp)

        if !isnothing(r)
            push!(T2_indices, k)
        end
    end

    active_systems_T2 = collect( (LUT[i].compound, LUT[i].spin_sys) for i in T2_indices )
    
    return SIG.SubsetVars(
        SIG.SubsetVarsIndices(shift_indices, phase_indices, T2_indices),
        vcat(active_systems_sp, active_systems_sp, active_systems_T2),
    )
end

function getflatindices(S::SIG.SubsetVars)::Vector{Int}
    return vcat(S.indices.shift, S.indices.phase, S.indices.T2 )
end


############### misc.

function isallfinite(A::Vector)::Bool
    
    return all( isallfinite(A[i]) for i in eachindex(A) )
end
        
function isallfinite(x::T)::Bool where T <: Real
    return isfinite(x)
end
