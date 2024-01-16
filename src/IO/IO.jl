
function readJSON(load_folder::String, file_name::String)
    return JSON3.read(
        read(
            joinpath(load_folder, file_name),
        ),
    )
end

### load surrogate model from disk.
function loadclmodel(
    ::Type{T}, # this is the floating-point type returned by JSON3 and BSON. default to Float64 on AMD64 CPU architetures.
    folder_path::String;
    file_name::String = "model",
    )::CLModelContainer{T} where T <: Real
    
    # dict_Phys = readJSON(folder_path, "Phys.json")
    # Phys, molecule_entries = HAM.deserializephysicalparams(Dict(dict_Phys))

    # dict_As = readJSON(folder_path, "As.json")
    # As = HAM.deserializemixture(dict_As)

    # dict_Bs = readJSON(folder_path, "Bs.json")
    # ss_params_set, op_range_set, λ0 = SIG.deserializclproxies(dict_Bs)

    # dict_itp = BSON.load(
    #     joinpath(folder_path, "itp.bson"),
    # )
    # itp_samps2 = SIG.deserializitpsamples(dict_itp)

    # Bs, MSS = SIG.recoverclproxies(
    #     itp_samps2,
    #     ss_params_set,
    #     op_range_set,
    #     As,
    #     λ0,
    # )
    Phys, As, molecule_entries = deserialize(
        joinpath(folder_path, "$(file_name)_SH"),
    )

    Bs, MSS = deserialize(
        joinpath(folder_path, "$(file_name)_CL"),
    )

    # force load a default set of model parameters to initialize it.
    resetmodel!(MSS)

    return CLModelContainer(Phys, As, Bs, MSS, molecule_entries)
end

function loadfidmodel(
    ::Type{T}, # this is the floating-point type returned by JSON3 and BSON. default to Float64 on AMD64 CPU architetures.
    ::SIG.UseSharedT2,
    folder_path::String;
    file_name::String = "model",
    )::FIDModelContainer{T, SIG.SharedT2{T}} where T <: Real
    
    Phys, As, molecule_entries = deserialize(
        joinpath(folder_path, "$(file_name)_SH"),
    )

    Cs, MSS = deserialize(
        joinpath(folder_path, "$(file_name)_SharedT2_FID"),
    )
    # force load a default set of model parameters to initialize it.
    resetmodel!(MSS)

    return FIDModelContainer(Phys, As, Cs, MSS, molecule_entries)
end

function loadfidmodel(
    ::Type{T}, # this is the floating-point type returned by JSON3 and BSON. default to Float64 on AMD64 CPU architetures.
    ::SIG.UseCoherenceT2,
    folder_path::String;
    file_name::String = "model",
    )::FIDModelContainer{T, SIG.CoherenceT2{T}} where T <: Real
    
    Phys, As, molecule_entries = deserialize(
        joinpath(folder_path, "$(file_name)_SH"),
    )

    Cs, MSS = deserialize(
        joinpath(folder_path, "$(file_name)_CoherenceT2_FID"),
    )
    # force load a default set of model parameters to initialize it.
    resetmodel!(MSS)

    return FIDModelContainer(Phys, As, Cs, MSS, molecule_entries)
end

# to default parameters.
function resetmodel!(
    MSS::Union{SIG.CLMixtureSpinSys{T, ST, PT, T2T}, SIG.FIDMixtureSpinSys{T, ST, PT, T2T}}
    ) where {T, ST, PT, T2T}

    N_compounds = SIG.getNentries(MSS)
    model_params = SIG.MixtureModelParameters(MSS, ones(T, N_compounds))

    params_LUT = getparamsLUT(MSS)
    p_default = getdefaultparameter(T, params_LUT)
    SIG.importmodel!(model_params, p_default)

    return nothing
end

abstract type ExperimentTrait end
struct Bruker1D <: ExperimentTrait end

function loadexperiment(::Type{T}, folder_path::String; file_name::String = "experiment")::ExperimentContainer{T} where T <: Real

    return loadexperiment(T, Bruker1D(), folder_path; file_name = file_name)
end

function loadexperiment(
    ::Type{T}, # this is the floating-point type returned by JSON3 and BSON. default to Float64 on AMD64 CPU architetures.
    ::Bruker1D,
    folder_path::String;
    file_name::String = "experiment",
    )::ExperimentContainer{T} where T <: Real

    return ExperimentContainer(loadBruker1Dexperiment(T, folder_path, file_name))
end

function loadBruker1Dexperiment(::Type{T}, load_path::String, file_name::String)::DSU.OutputContainer1D{T, DSU.Bruker1D1HSettings{T}} where T <: Real
    return deserialize(joinpath(load_path, file_name))
end

# # pre-processing via DSU.jl. It gets an estimate of the zero ppm peak, and truncates the first few samples of the time series to rid of the dead-time.
# function getpreprocessfunc(
#     params::ExperimentParameters{T,DT};
#     solvent_ppm_guess::T = convert(T, 4.7),
#     solvent_window_ppm::T = convert(T, 0.1),
#     λ_lower::T = convert(T, 0.7),
#     λ_upper::T = convert(T, 20.0),
#     ) where {T <: AbstractFloat, DT}

#     s_t, S, hz2ppmfunc, ppm2hzfunc, ν_0ppm, α_0ppm, β_0ppm, λ_0ppm, Ω_0ppm,
#         α_solvent, β_solvent, λ_solvent, Ω_solvent, results_0ppm,
#         results_solvent = DSU.loadspectrum(
#         params.data;
#         N_real_numbers_FID_data = params.TD, # this should be equal to length(data)/2
#         spectral_width_ppm = params.SW,
#         carrier_frequency_Hz = params.SFO1,
#         carrier_frequency_offset_Hz = params.O1,
#         fs_Hz = params.fs,
#         solvent_ppm = solvent_ppm_guess,
#         solvent_window_ppm = solvent_window_ppm,
#         λ_lower = λ_lower,
#         λ_upper = λ_upper,
#     )

#     return s_t, ν_0ppm, α_0ppm, β_0ppm, λ_0ppm
# end