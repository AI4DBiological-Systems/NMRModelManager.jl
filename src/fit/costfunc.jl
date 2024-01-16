##### scale the optimization variables, s.t. optim_var = param / scale.
# no longer used for gradient based optim, but will use for Bayesian opt.

# for all variables, or subset of continugous variables.
struct VariableRanges
    shift::UnitRange{Int}
    phase::UnitRange{Int}
    T2::UnitRange{Int}
end

function VariableRanges(params_mapping::SIG.ParamsMapping)::VariableRanges
    return VariableRanges(
        params_mapping.shift.st[begin][begin]:params_mapping.shift.fin[end][end],
        params_mapping.phase.st[begin][begin]:params_mapping.phase.fin[end][end],
        params_mapping.T2.st[begin][begin]:params_mapping.T2.fin[end][end],
    )
end

struct VariableScaling{T}
    shift::T
    phase::T
    T2::T
end

# the return struct should have field names: shift, phase, T2, that are ranges (VariableRanges) or Vector{Int} (SIG.SubsetVars).
function getvarmapping(info::SIG.SubsetVars, args...)::SIG.SubsetVarsIndices
    return info.indices
end

function getvarmapping(::SIG.AllVars, MSS::SIG.MixtureSpinSys)::VariableRanges
    return VariableRanges(SIG.ParamsMapping(MSS))
end




############### front end to costfunc setup.

# default to optimize all parameters of the model.
# resets model.
function setupl2cost!(
    MSS::SIG.CLMixtureSpinSys,
    y::Vector{Complex{T}},
    U, # in Hz.
    w_lbs::Vector{T},
    w_ubs::Vector{T},
    ) where T

    resetmodel!(MSS)

    return setupl2cost!(
        SIG.AllVars(),
        Vector{T}(undef, 0),
        MSS,
        y,
        U,
        w_lbs,
        w_ubs,
    )
end

# does not reset model to p_base. If p_base is empty, does not modify model.
# allow for optimizing only a subset of all variables, but need to provide a p_base.
function setupl2cost!(
    info::SIG.VariableSetTrait,
    p_base::Vector{T}, # only used if typeof(info) <: SIG.SubsetVars. contains the constant parameters of the subset variable model.
    MSS::SIG.CLMixtureSpinSys,
    y::Vector{Complex{T}},
    U, # in Hz.
    w_lbs::Vector{T},
    w_ubs::Vector{T},
    ) where T

    N = SIG.getNentries(MSS)
    w = ones(T, N)
    model_params = SIG.MixtureModelParameters(MSS, w)

    C, BLS_params = setupfitpreliminaries(
        MSS, y, U, w_lbs, w_ubs,
    )

    # If we've a subset variable model, the following set ups the constant parameters in `p_base`.
    #preparesubsetmodel!(info, model_params, BLS_params, C, p_base, y)
    if !isempty(p_base)
        SIG.importmodel!(model_params, p_base)
    end

    f = pp->evalenvelopecost!(
        model_params,
        BLS_params,
        C, 
        pp,
        y,
        info,
    )

    fdf! = (gg,pp)->evalenvelopegradient!(
        gg,
        model_params,
        BLS_params,
        C,
        pp,
        y,
        info,
    )

    df! = fdf! # TODO implement an efficient version that computes df!, not f.

    return f, df!, fdf!, w, C #, BLS_params, C
end


#################


function setupfitpreliminaries(
    MSS,
    y::Vector{Complex{T}},
    U,
    w_lbs::Vector{T},
    w_ubs::Vector{T};
    BLS_config::BLSConfig{T} = BLSConfig{T}(),
    ) where T

    U_rad = U .* twopi(T)
    
    #
    shifts, phases, T2s = MSS.shifts, MSS.phases, MSS.T2s
    mapping = SIG.ParamsMapping(shifts, phases, T2s)

    # data container.
    X, gs_re, gs_im, shift_multi_inds, phase_multi_inds,
    T2_multi_inds = costfuncsetup(mapping, MSS, U_rad) # costfuncsetup() takes U in radians.

    C = CostFuncBuffer(
        X, gs_re, gs_im, shift_multi_inds, phase_multi_inds, T2_multi_inds,
    )
    
    #
    BLS_params = setupwsolver(X, MSS, w_lbs, w_ubs, y, BLS_config)
    fill!(BLS_params.primal_initial, zero(T))

    return C, BLS_params
end

function fetchbounds(
    Ws::Vector{MT},
    MSS::SIG.MixtureSpinSys;
    shift_proportion::T = convert(T, 0.9),
    phase_lb::T = convert(T, -π),
    phase_ub::T = convert(T, π),
    )::Tuple{Vector{T},Vector{T}} where {T, MT <: SIG.MoleculeType}

    N = SIG.getNentries(MSS)
    w = ones(T, N)
    model_params = SIG.MixtureModelParameters(MSS, w)
    lbs0, ubs0 = SIG.fetchbounds(
        model_params,
        Ws;
        shift_proportion = shift_proportion,
        phase_lb = phase_lb,
        phase_ub = phase_ub,
    )
    lbs = lbs0
    ubs = ubs0

    return lbs, ubs
end

function getcostdata(
    model::ModelContainer,
    experiment::ExperimentContainer{T},
    config::DataObjectiveConfig{T},
    ) where T <: Real
    
    As, molecule_entries = model.As, model.molecule_entries

    #s_t = experiment.s_t
    fs, SW, ν_0ppm = experiment.fs, experiment.SW, experiment.ν_0ppm
    y, U_y = experiment.y, experiment.U_y

    hz2ppmfunc = uu->(uu - ν_0ppm)*SW/fs
    #ppm2hzfunc = pp->(ν_0ppm + pp*fs/SW)

    Δcs_padding = config.Δcs_padding

    # # Specify region for fit.
    # get intervals.
    ΩS0 = getΩS(As)
    ΩS0_ppm = getPs(ΩS0, hz2ppmfunc)

    Δsys_cs = initializeΔsyscs(As, Δcs_padding)
    exp_info = NMRSpecifyRegions.setupexperimentresults(
        molecule_entries,
        ΩS0_ppm,
        Δsys_cs;
        min_dist = Δcs_padding,
    )

    # get cost.
    P_y = hz2ppmfunc.(U_y)
    cost_inds, band_inds_set = NMRSpecifyRegions.getcostinds(exp_info, P_y) # cost_inds is the union of band_inds_set, then kept unique inds.
    cost_inds = sort(cost_inds)
    
    return packagecostdata(band_inds_set, cost_inds, U_y, y)
end

function packagecostdata(
    P_bounds::Vector{Tuple{T,T}},
    freq_params::FrequencyParameters{T},
    #cost_inds::Vector{Int},
    U_y::Vector{T}, # already sorted.
    y::Vector{Complex{T}},
    ) where T <: AbstractFloat

    @assert issorted(U_y)

    hz2ppmfunc = uu->hz2ppm(uu, freq_params)
    #ppm2hzfunc = uu->ppm2hz(uu, freq_params)
    P_y = hz2ppmfunc.(U_y)

    band_inds_set = collect(
        findall(
            xx->(P_bounds[m][begin] <= xx <= P_bounds[m][end]),
            P_y,
        )
        for m in eachindex(P_bounds)
    )

    # manually do what NMRSpecifyRegions.getcostinds is doing, also put in sorted inds.
    cost_inds = sort(unique(collect(Iterators.flatten(band_inds_set))))
    
    return packagecostdata(band_inds_set, cost_inds, U_y, y)
end

function packagecostdata(
    band_inds_set::Vector{Vector{Int}},
    cost_inds::Vector{Int},
    U_y::Vector{T},
    y::Vector{Complex{T}},
    ) where T <: AbstractFloat

    U_cost = U_y[cost_inds]
    #P_cost = P_y[band_inds]

    y_cost = y[cost_inds]

    # new idea for normalization.
    scale_factor = one(T)/maximum(abs.(y_cost))
    y = y .* scale_factor
    y_cost = y_cost .* scale_factor

    return FitData(
        scale_factor,
        cost_inds,
        band_inds_set,
        y_cost,
        U_cost,
        collect( y[band_inds_set[r]] for r in eachindex(band_inds_set)),
        collect( U_y[band_inds_set[r]] for r in eachindex(band_inds_set)),
    ) #U_cost, y_cost, band_inds, band_inds_set, exp2fit_scale_factor
end


###### specify interval.
function getΩS(As::Vector{HAM.SHType{T}}) where T

    ΩS = Vector{Vector{Vector{T}}}(undef, length(As))

    for n in eachindex(As)

        ΩS[n] = Vector{Vector{T}}(undef, length(As[n].Ωs))
        for i in eachindex(As[n].Ωs)

            ΩS[n][i] = copy(As[n].Ωs[i])

        end
    end

    return ΩS
end

function getPs( ΩS::Vector{Vector{Vector{T}}}, hz2ppmfunc) where T <: Real

    N_compounds = length(ΩS)

    Ps = Vector{Vector{Vector{T}}}(undef, N_compounds)
    for n = 1:N_compounds

        Ps[n] = Vector{Vector{T}}(undef, length(ΩS[n]))
        for i in eachindex(ΩS[n])

            Ps[n][i] = Vector{T}(undef, length(ΩS[n][i]))
            for l in eachindex(ΩS[n][i])

                Ps[n][i][l] = hz2ppmfunc( ΩS[n][i][l]/twopi(T) )
            end
        end
    end

    return Ps
end

function getPsnospininfo(As::Vector{HAM.SHType{T}}, hz2ppmfunc) where T

    ΩS_ppm = Vector{Vector{T}}(undef, length(As))

    for (n,A) in enumerate(As)

        ΩS_ppm[n] = hz2ppmfunc.( combinevectors(A.Ωs) ./ twopi(T) )

    end

    return ΩS_ppm
end

function initializeΔsyscs(As, x::T) where T

    Δsys_cs = Vector{Vector{T}}(undef,  length(As))

    for n in eachindex(As)

        N_sys = length(As[n].N_spins_sys)
        Δsys_cs[n] = Vector{T}(undef, N_sys)

        for i in eachindex(Δsys_cs[n])
            Δsys_cs[n][i] = x
        end

        # for _ in eachindex(As[n].αs_singlets)
        #     push!(Δsys_cs[n], x)
        # end
    end

    return Δsys_cs
end
