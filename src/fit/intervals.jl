# identifying isolated nuclei in an frequency interval from the cost func.

struct BandsInfo{T}
    cs_flat::Vector{T} # all the reference chemical shifts of the model shift parameters.
    inds::Vector{Vector{Int}} # [interval index][index wrt `cs`]. same length as band_inds.
    var_subsets::Vector{SIG.SubsetVars} # a set of subsets of model parameters. Each subset is for an interval.

    # parameters used for determining the isolated variables.
    Δcs_lb::T
    Δcs_ub::T
    N_cs_threshold::Int # bands with this number of nuclei or fewer are considered isolated.

    # the isolated variables.
    isolated_inds::Vector{Int} # cs_inds[isolated_inds] are the intervals that have less than N_cs_threshold number of nuclei.
end

function getNvars(A::BandsInfo)::Int
    N_cs = length(A.cs_flat)
    N_spin_sys = length(A.inds)

    return N_cs*2 + N_spin_sys # assuming CoherenceShift, CoherencePhse, which is N_cs each, and SHaredT2, which is N_spin_sys.
end

struct DataIntervalInfo{T}
    fit_data::FitData{T}
    info::BandsInfo{T}
end

############# schedule jobs for isolated intervals, to reduce problem complexity.
function identifyisolatedintervals(
    model::CLModelContainer,
    freq_params::FrequencyParameters,
    experiment::ExperimentContainer{T},
    fit_data::FitData{T};
    Δcs_lb::T = convert(T, 0.08),
    Δcs_ub::T = convert(T, 0.08),
    N_cs_threshold::Int = 3, # maximum number of chemical shifts for a returned interval.
    )::DataIntervalInfo{T} where T <: AbstractFloat

    U_data, band_inds = experiment.U_y, fit_data.band_inds

    params_LUT = getparamsLUT(model)

    # ## create a subset vars based on an interval.
    cs = getcs(model.Phys)
    band_cs_inds = identifyshifts(band_inds, U_data, cs, freq_params)

    # store the variable indices information for every interval.
    var_subsets =  Vector{SIG.SubsetVars}(undef, 0)
    for m in eachindex(band_cs_inds) 
        s = findvars(params_LUT, band_cs_inds[m])
        push!(var_subsets, s)
    end
    
    # intervals with low number of chemical shifts: only consider intervals that have the same chemical shifts under a perturbation of Δcs_lb and Δcs_ub.
    cs_lb_inds = identifyshifts(band_inds, U_data, cs .- Δcs_lb, freq_params)
    cs_ub_inds = identifyshifts(band_inds, U_data, cs .+ Δcs_ub, freq_params)
    has_constant_cs = cs_lb_inds .== cs_ub_inds

    # assemble intervals that have less/equal to `N_cs_threshold` number of chemical shifts.
    isolated_bands = Vector{Int}(undef, 0)

    for m in eachindex(has_constant_cs)
        if has_constant_cs[m] && length(band_cs_inds[m]) <= N_cs_threshold
            
            push!(isolated_bands, m)
        end
    end

    return DataIntervalInfo(
        fit_data,
        BandsInfo(
            cs,
            band_cs_inds,
            var_subsets,
            Δcs_lb,
            Δcs_ub,
            N_cs_threshold,
            isolated_bands,
        ),
    )
end

############# methods for working with objective functions on an interval.


# for updating the full set of variables given values of a subset.
# the logic is similar to reducebounds!().
function updatefull!(
    p_full::Vector{T},
    p::Vector{T},
    subset_vars::SIG.SubsetVars,
    ) where T
    
    inds = getflatindices(subset_vars)

    return updatearray!(p_full, p, inds)
end

function updatearray!(
    out::Vector{T}, # mutates.
    x::Vector{T},
    inds::Vector{Int},
    )::Nothing where T <: AbstractFloat

    @assert length(inds) <= length(out)

    k = 0
    for i in inds

        k+= 1
        out[i] = x[k]
    end

    return nothing
end

# reduce the bounds. TODO use a non-allocating version.
function reducebounds!(
    lbs::Vector{T}, # mutates.
    ubs::Vector{T}, # mutates.
    inds::Vector{Int},
    p_ref::Vector{T};
    gap::T = convert(T, 1e-6),
    )::Nothing where T <: AbstractFloat

    @assert length(lbs) == length(ubs) == length(p_ref)

    for i in inds
        lbs[i] = p_ref[i] - gap
        ubs[i] = p_ref[i] + gap
    end

    return nothing
end

# reduce bounds for variables that are in subset_vars. TODO use a non-allocating version.
function reducebounds!(
    lbs::Vector{T}, # mutates.
    ubs::Vector{T}, # mutates.
    subset_vars::SIG.SubsetVars,
    p_ref::Vector{T};
    gap::T = convert(T, 1e-6),
    )::Nothing where T <: AbstractFloat

    inds = getflatindices(subset_vars)
    
    return reducebounds!(lbs, ubs, inds, p_ref; gap = gap)
end

# reduce bounds for variables that are not in subset_vars.
function reduceboundscomplement!(
    lbs::Vector{T}, # mutates.
    ubs::Vector{T}, # mutates.
    subset_vars::SIG.SubsetVars,
    p_ref::Vector{T};
    gap::T = convert(T, 1e-6),
    )::Nothing where T <: AbstractFloat

    inds = getflatindices(subset_vars)
    comp_inds = setdiff(1:length(p_ref), inds)
    
    if !isempty(comp_inds)
        return reducebounds!(lbs, ubs, comp_inds, p_ref; gap = gap)
    end

    return nothing
end

function getcomplementinds(MSS::SIG.MixtureModelParameters, subset_vars::SIG.SubsetVars)::Vector{Int}
    
    inds = getflatindices(subset_vars)

    N_vars = SIG.getNvars(MSS)
    comp_inds = setdiff(1:N_vars, inds)
    
    return comp_inds
end

