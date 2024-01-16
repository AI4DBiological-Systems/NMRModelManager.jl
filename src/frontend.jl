
# contains the mutable objects or traits that accompanies SetupConfigCL.
struct FitState{T <: AbstractFloat, PT <: Union{Vector{Tuple{T,T}}, Nothing}}
    
    # actual bounds. Used for lock-ins.
    w_lbs::Vector{T}
    w_ubs::Vector{T}
    lbs::Vector{T}
    ubs::Vector{T}

    interval_bounds_ppm::PT # if Vector{Tuple{T,T}}, then or each entry, first is lb, second is ub. In units ppm.
end

function FitState(::Type{T})::FitState{T,Nothing} where T <: AbstractFloat
    return FitState(
        Vector{T}(undef, 0), Vector{T}(undef, 0), Vector{T}(undef, 0), Vector{T}(undef, 0), nothing,
    )
end

function FitState(P_bounds::Vector{Tuple{T,T}})::FitState{T,Vector{Tuple{T,T}}} where T
    return FitState(
        Vector{T}(undef, 0), Vector{T}(undef, 0), Vector{T}(undef, 0), Vector{T}(undef, 0), P_bounds,
    )
end

function resetconfigbounds!(C::FitState)
    
    resize!(C.lbs, 0)
    resize!(C.ubs, 0)
    resize!(C.w_lbs, 0)
    resize!(C.w_ubs, 0)

    return nothing
end

function initializeconfigbounds!(
    S::FitState,
    config::SetupConfigCore{T},
    model::CLModelContainer{T},
    ) where T

    lbs, ubs = fetchbounds(
        model.Bs,
        model.MSS;
        shift_proportion = config.shift_proportion,
        phase_lb = config.phase_lb,
        phase_ub = config.phase_ub,
    )
    
    resize!(S.lbs, length(lbs))
    S.lbs[:] = lbs

    resize!(S.ubs, length(ubs))
    S.ubs[:] = ubs

    N = getNentries(model)
    resize!(S.w_lbs, N)
    fill!(S.w_lbs, config.default_w_lb)

    resize!(S.w_ubs, N)
    fill!(S.w_ubs, config.default_w_ub)

    return nothing
end


abstract type ObjectiveContainer end

struct l2ObjectiveContainer{T <: AbstractFloat, ST <: SubArray, F <: Function, DF <: Function, FDF <: Function} <: ObjectiveContainer
    f::F
    df!::DF
    fdf!::FDF

    # buffers.
    w::Vector{T}
    cost_buffer::CostFuncBuffer{T,ST}
end

struct CLFitContainer{T <: AbstractFloat, CT <: ObjectiveContainer}
    model::CLModelContainer{T}
    experiment::ExperimentContainer{T}
    intervals::DataIntervalInfo{T}
    lbs::Vector{T}
    ubs::Vector{T}
    
    objective::CT
    interval_objectives::Vector{CT}
end

function getfitdata(A::CLFitContainer{T})::FitData{T} where T
    return A.intervals.fit_data
end

abstract type FitOption end
struct Usel2cost <: FitOption end


# assumes the config bounds are initialized.
function setupfit(
    fit_option::FitOption,
    state::FitState{T, Nothing},
    experiment::ExperimentContainer{T},
    model::CLModelContainer{T},
    config::SetupConfigCL{T},
    ) where T

    fit_data = getcostdata(model, experiment, config.core.data_objective)

    return setupfit(fit_option, state, fit_data, experiment, model, config)
end

# assumes the config bounds are initialized.
function setupfit(
    ::Usel2cost,
    state::FitState,
    fit_data::FitData{T},
    experiment::ExperimentContainer{T},
    model::CLModelContainer{T},
    config::SetupConfigCL{T},
    ) where T
    
    MSS = model.MSS
    fs, SW, ν0 = experiment.fs, experiment.SW, experiment.ν_0ppm
    freq_params = FrequencyParameters(fs, SW, ν0)

    lbs, ubs = state.lbs, state.ubs

    intervals = identifyisolatedintervals(
        model,
        freq_params,
        experiment,
        fit_data;
        N_cs_threshold = config.core.N_cs_threshold,
    )

    objective = createobjectivecontainer(
        Usel2cost(),
        MSS,
        fit_data.y_cost,
        fit_data.U_cost,
        state,
    )

    interval_objectives = collect(
        createobjectivecontainer(
            Usel2cost(),
            MSS,
            fit_data.y_cost_bands[r],
            fit_data.U_cost_bands[r],
            state,
        )
        for r in eachindex(fit_data.band_inds)
    )

    return CLFitContainer(
        model, experiment, intervals, lbs, ubs, objective, interval_objectives,
    )
end

# assumes the config bounds are initialized.
# fontend, manual version: the cost intervals are specified in P_bounds.
function setupfit(
    fit_option::FitOption,
    state::FitState{T,Vector{Tuple{T,T}}},
    experiment::ExperimentContainer{T},
    model::CLModelContainer{T},
    config::SetupConfigCL{T},
    ) where T

    P_bounds = state.interval_bounds_ppm
    @assert !isempty(P_bounds)

    #fit_data_prelim = getcostdata(model, experiment, config.data_objective)

    fs, SW, ν0 = experiment.fs, experiment.SW, experiment.ν_0ppm
    freq_params = FrequencyParameters(fs, SW, ν0)

    fit_data = packagecostdata(
        P_bounds,
        freq_params,
        #fit_data_prelim.cost_inds,
        experiment.U_y,
        experiment.y,
    )

    return setupfit(fit_option, state, fit_data, experiment, model, config)
end



function createobjectivecontainer(
    ::Usel2cost,
    MSS::SIG.CLMixtureSpinSys,
    y_cost::Vector{Complex{T}},
    U_cost::Vector{T},
    state::FitState,
    ) where T <: AbstractFloat

    f, df!, fdf!, w, C = setupl2cost!(
        MSS,
        y_cost,
        U_cost,
        state.w_lbs,
        state.w_ubs,
    )

    return l2ObjectiveContainer(f, df!, fdf!, w, C)
end

function getdefaultparameter(F::CLFitContainer{T, CT})::Vector{T} where {T, CT}

    params_LUT = getparamsLUT(F.model)
    p_default = getdefaultparameter(T, params_LUT)

    return p_default
end


# used for re-simulation given Phys.


### relative error.

# l-2 objective means the objective itself is the absolute error.
# if r is a valid index in F.intervals.band_inds, compute error over that interval.
# otherwise, compute error over full cost function
function evalfiterror(
    F::CLFitContainer{T,CT},
    p_test::Vector{T},
    r::Int,
    ) where {T <: AbstractFloat, CT <: l2ObjectiveContainer}

    fit_data = getfitdata(F)

    # p_test must be the full set of variables.
    @assert length(F.lbs) == length(p_test)

    if 1 <= r <= length(fit_data.band_inds)

        abs_err = sqrt(F.interval_objectives[r].f(p_test))

        y = fit_data.y_cost_bands[r]
        rel_err = abs_err/norm(y)
        
        return abs_err, rel_err
    end

    abs_err = sqrt(F.objective.f(p_test))
    rel_err = abs_err/norm(fit_data.y_cost)

    return abs_err, rel_err
end
