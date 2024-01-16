

####### container types


# struct FullModelContainer{T}
#     Phys::Vector{PhysicalParamsType{T}}
#     As::Vector{SHType{T}}
#     Bs::Vector{MoleculeType{T, SpinSysParams{CoherenceShift{T}, CoherencePhase{T}, SharedT2{T}}, SIG.CLOperationRange{T}}}
#     Cs::Vector{MoleculeType{T, SpinSysParams{CoherenceShift{T}, CoherencePhase{T}, SharedT2{T}}, SIG.FIDOperationRange{T}}}
#     MSS_cl::CLMixtureSpinSys{T, CoherenceShift{T}, CoherencePhase{T}, SharedT2{T}}
#     MSS_fid::FIDMixtureSpinSys{T, CoherenceShift{T}, CoherencePhase{T}, SharedT2{T}}
#     molecule_entries::Vector{String}
# end

abstract type ModelContainer end

struct CLModelContainer{T} <: ModelContainer
    Phys::Vector{PhysicalParamsType{T}}
    As::Vector{SHType{T}}
    Bs::Vector{MoleculeType{T, SpinSysParams{CoherenceShift{T}, CoherencePhase{T}, SharedT2{T}}, SIG.CLOperationRange{T}}}
    MSS::CLMixtureSpinSys{T, CoherenceShift{T}, CoherencePhase{T}, SharedT2{T}}
    molecule_entries::Vector{String}
end

struct FIDModelContainer{T, T2T} <: ModelContainer
    Phys::Vector{PhysicalParamsType{T}}
    As::Vector{SHType{T}}
    Cs::Vector{MoleculeType{T, SpinSysParams{CoherenceShift{T}, CoherencePhase{T}, T2T}, SIG.FIDOperationRange{T}}}
    MSS::FIDMixtureSpinSys{T, CoherenceShift{T}, CoherencePhase{T}, T2T}
    molecule_entries::Vector{String}
end

# function getmapping(model::ModelContainer{T}) where T <: Real
#     return getmapping(model.MSS)
# end

abstract type FrequencyConversionType end

struct ExperimentContainer{T} <: FrequencyConversionType
    y::Vector{Complex{T}}
    U_y::Vector{T}
    fs::T
    SW::T
    ν_0ppm::T
    α_0ppm::T
    β_0ppm::T
    λ_0ppm::T
end

function ExperimentContainer(A::DSU.OutputContainer1D{T,ST})::ExperimentContainer{T} where {T, ST}
    tmp = DSU.FrequencyConversionParameters(A)
    return ExperimentContainer(
        A.spectrum.y,
        A.spectrum.U_y,
        tmp.fs,
        tmp.SW,
        tmp.ν_0ppm,
        A.freq_ref.α,
        A.freq_ref.β,
        A.freq_ref.λ,
    )
end

##### conversions

# same as DSU.FrequencyConversionParameters, but create new type here to allow subtyping for future expansion.
struct FrequencyParameters{T} <: FrequencyConversionType
    fs::T
    SW::T
    ν_0ppm::T
end

function FrequencyParameters(A::ExperimentContainer{T})::FrequencyParameters{T} where T
    return FrequencyParameters(A.fs, A.SW, A.ν_0ppm)
end

# hz2ppmfunc = uu->hz2ppm(uu, experiment)
function hz2ppm(u::T, A::FrequencyConversionType)::T where T
    return (u - A.ν_0ppm)*A.SW/A.fs
end

# ppm2hzfunc = pp->ppm2hz(pp, experiment)
function ppm2hz(p::T, A::FrequencyConversionType)::T where T
    return (A.ν_0ppm + p*A.fs/A.SW)
end

# ppm2radfunc = pp->ppm2hzfunc(pp)*2*π
function ppm2rad(p::T, A::FrequencyConversionType)::T where T
    return ppm2hz(p, A)*twopi(T)
end

# rad2ppmfunc = ww->hz2ppmfunc(ww/(2*π))
function rad2ppm(ω::T, A::FrequencyConversionType)::T where T
    return hz2ppm(ω/twopi(T), A)
end

function Δppm2Δrad(Δp::T, A::FrequencyConversionType)::T where T
    return ppm2rad(Δp, A) - ppm2rad(zero(T), A)
end

# inverse of Δppm2Δrad().
function Δrad2Δppm(Δω::T, A::FrequencyConversionType)::T where T
    return rad2ppm(Δω + ppm2rad(zero(T), A), A)
end


############## configs

@kwdef struct DataObjectiveConfig{T}
    #offset_ppm::T = convert(T, 0.3) # 0.3 #in units ppm.
    Δcs_padding::T = convert(T, 0.1) # 0.02 or 0.1 #in units ppm.
end


struct FitData{T<:AbstractFloat}
    scale_factor::T
    cost_inds::Vector{Int}
    band_inds::Vector{Vector{Int}}

    # y_cost = (experiment.y[cost_inds] .* scale_factor)
    y_cost::Vector{Complex{T}}
    U_cost::Vector{T}

    y_cost_bands::Vector{Vector{Complex{T}}}
    U_cost_bands::Vector{Vector{T}}
end


@kwdef struct SetupConfigCore{T}
    resonance::HAM.SHConfig{T} = HAM.SHConfig{T}(
        coherence_tol = convert(T, 0.01),
        relative_α_threshold = convert(T, 0.005),
        max_deviation_from_mean = convert(T, 0.2), # positive number. Larger means less resonance groups, but worse approximation for each group.
        acceptance_factor = convert(T, 0.99), # keep this close to 1. Takes values from (0,1).
        total_α_threshold = convert(T, 0.01), # final intensity pruning.
    )

    unique_cs_digits::Int = 6

    # objective.
    data_objective::DataObjectiveConfig{T} = DataObjectiveConfig{T}()
    
    # default bounds.
    phase_lb = zero(T) #convert(T, -Inf)
    phase_ub = twopi(T) #convert(T, Inf)
    shift_proportion = convert(T, 0.9)

    # For reset or initialization.
    default_w_lb = zero(T)
    default_w_ub = convert(T, 10.0)

    # intervals.
    N_cs_threshold::Int = 3

    # diagnostics
    Δc_valid_atol::T = convert(T, 0.1)
    coherence_sum_zero_tol::T = convert(T, 1e-2)
end

abstract type SetupConfig end

@kwdef struct SetupConfigCL{T} <: SetupConfig

    core::SetupConfigCore{T} = SetupConfigCore{T}()
    
    surrogate = SIG.CLSurrogateConfig{T}( 
        Δr = convert(T, 1.0), # radial frequency resolution: smaller means slower to build surrogate, but more accurate.
        Δκ_λ = convert(T, 0.05), # T2 multiplier resolution. smaller means slower to build surrogate, but more accurate.
        Δcs_max_scalar = convert(T, 0.2), # In units of ppm. interpolation border that is added to the lowest and highest resonance frequency component of the mixture being simulated.
        κ_λ_lb = convert(T, 0.5), # lower limit for κ_λ for which the surrogate is made from.
        κ_λ_ub = convert(T, 2.5), # upper limit for κ_λ for which the surrogate is made from.
        ppm_padding = convert(T , 0.5),
    )
end

@kwdef struct SetupConfigFID{T} <: SetupConfig

    core::SetupConfigCore{T} = SetupConfigCore{T}()

    surrogate = SIG.FIDSurrogateConfig{T}(
        Δt = convert(T, 1e-5),
        t_lb = zero(T),
        t_ub = convert(T, 3.0),
    )
end