
# container to identify each experiment.
struct ExperimentInfo
    experiment_full_path::String # file path to experiment folder.
    project_name::String # label for this experiment.
    molecule_entries::Vector{String} # what molecules are in this experiment.
    source_tag::String # notes about the source of this data.
    #sizes_of_spin_systems::Vector{Vector{Int}} # [molecule][spin system]
end

######## load experiment.


# struct ExperimentParameters{T<:AbstractFloat, DT}
#     dic::DT # dictionary type, possibly from Python.
#     data::Vector{Complex{T}}
#     TD::Int # number of raw data points.
#     SW::T # spectral window, in ppm.
#     SFO1::T # spectrometer frequency in MHz.
#     O1::T # carrier frequency in Hz.
#     fs::T # sampling frequency for each complex number in `data`, in Hz.
# end

# function ExperimentParameters(
#     ::Type{T},
#     dic::DT;
#     data::Vector{Complex{T}} = convert(Vector{Complex{T}}, NaN),
#     TD::Int = 0,
#     SW::T = convert(T, NaN),
#     SFO1::T = convert(T, NaN),
#     O1::T = convert(T, NaN),
#     fs::T = convert(T, NaN),
#     ) where {T,DT}

#     return ExperimentParameters(dic, data, TD, SW, SFO1, O1, fs)
# end


## this part goes into example scripts.

### end goes to example scripts.

function storeexperimentsetup(
    SH_config::HAM.SHConfig{T},
    cl_proxy_config::SIG.CLSurrogateConfig{T},
    fid_proxy_config::SIG.FIDSurrogateConfig{T},
    infos::Vector{ExperimentInfo},

    save_root_folder::String,
    H_params_path::String, # joinpath(root_data_path, "coupling_info"), # folder of coupling values. # replace with your own values in actual usage.
    molecule_mapping_file_path::String; # joinpath(root_data_path, "molecule_name_mapping"), joinpath(molecule_mapping_root_path, "select_molecules.json");
    freq_ref_config::DSU.FitSingletConfig{T} = DSU.FitSingletConfig{T}(
    
        frequency_center_ppm = zero(T), # the singlet is at 0 ppm.
        frequency_radius_ppm = convert(T, 0.3),

        # smaller λ means sharper line shape at the resonance frequency, less heavy tails.
        λ_lb = convert(T, 0.2),
        λ_ub = convert(T, 7.0),
    ),
    solvent_config::Union{Nothing, DSU.FitSingletConfig{T}} = DSU.FitSingletConfig{T}(

        frequency_center_ppm = convert(T, 4.75), # in this example, we assume the solvent is between 4.75 - 0.15 to 4.75 + 0.15 ppm.
        frequency_radius_ppm = convert(T, 0.15),
        
        # smaller λ means sharper line shape at the resonance frequency, less heavy tails.
        λ_lb = convert(T, 1e-3),
        λ_ub = convert(T, 20.0),
    ),
    data_config::DSU.SetupConfig{T} = DSU.SetupConfig{T}(
        rescaled_max = convert(T, 10.0),
        max_CAR_ν_0ppm_difference_ppm = convert(T, 0.4), # used to assess whether the experiment is missing (or has a very low intensity) 0 ppm peak.
        offset_ppm = convert(T, 0.5),
        FID_name = "fid",
        settings_file_name = "acqu",
    ),
    fallback_λ0::T = convert(T, 3.4),
    unique_cs_digits::Int = 6,
    Δc_valid_atol = convert(T, 0.1), # diagnostics. Set a larger number if it keeps failing.
    coherence_sum_zero_tol = convert(T, 1e-2), # diagnostics. Set a larger number if it keeps failing.
    ) where T <: AbstractFloat

    for info in infos
        storeexperimentsetup(
            SH_config,
            cl_proxy_config,
            fid_proxy_config,
            info,
            save_root_folder,
            H_params_path,
            molecule_mapping_file_path,
            freq_ref_config,
            solvent_config,
            data_config;
            fallback_λ0 = fallback_λ0,
            unique_cs_digits = unique_cs_digits,
            Δc_valid_atol = Δc_valid_atol,
            coherence_sum_zero_tol = coherence_sum_zero_tol,
        )
    end

    return nothing
end

# this is not meant for long-term storage of data types on disk.
# This means compatibility between Julia versions or package versions are not guaranteed .
function storeexperimentsetup(
    SH_config::HAM.SHConfig{T},
    cl_proxy_config::SIG.CLSurrogateConfig{T},
    fid_proxy_config::SIG.FIDSurrogateConfig{T},
    info::ExperimentInfo,
    save_root_folder::String,
    H_params_path::String, # joinpath(root_data_path, "coupling_info"), # folder of coupling values. # replace with your own values in actual usage.
    molecule_mapping_file_path::String, # joinpath(root_data_path, "molecule_name_mapping"), joinpath(molecule_mapping_root_path, "select_molecules.json");
    freq_ref_config::DSU.FitSingletConfig{T},
    solvent_config::Union{Nothing, DSU.FitSingletConfig{T}},
    data_config;
    fallback_λ0::T = convert(T, 3.4),
    unique_cs_digits::Int = 6,
    Δc_valid_atol::T = convert(T, 0.1), # diagnostics. Set a larger number if it keeps failing.
    coherence_sum_zero_tol::T = convert(T, 1e-2), # diagnostics. Set a larger number if it keeps failing.
    ) where T <: AbstractFloat

    experiment_full_path, project_name, molecule_entries = info.experiment_full_path, info.project_name, info.molecule_entries

    #println("Working on $project_name")

    loaded_experiment = DSU.setupBruker1Dspectrum(
        T,
        experiment_full_path,
        data_config;
        freq_ref_config = freq_ref_config,
        solvent_config = solvent_config,
    )
    #data, spectrum, freq_ref, solvent = DSU.unpackcontainer(loaded_experiment)
    freq_ref = loaded_experiment.freq_ref
    freq_params_DSU = DSU.FrequencyConversionParameters(loaded_experiment)
    
    # duplicated identical data structures. TODO consider unifying MG.FrequencyParameters with DSU.FrequencyConversionParameters
    fs, SW, ν0 = freq_params_DSU.fs, freq_params_DSU.SW, freq_params_DSU.ν_0ppm
    freq_params = FrequencyParameters(fs, SW, ν0)

    ####### simulate.

    λ0 = freq_ref.λ
    if !(freq_ref_config.λ_lb <= λ0 <= freq_ref_config.λ_ub)
        println("Warning: λ_0ppm is $λ_0ppm, which is outisde of [$(freq_ref_config.λ_lb), $(freq_ref_config.λ_ub)]. Using the default λ0 value, which is $(fallback_λ0).")
        λ0 = fallback_λ0
    end

    # save.
    save_folder_path = joinpath(save_root_folder, project_name)
    isdir(save_folder_path) || mkpath(save_folder_path); # make save folder if it doesn't exist.

    # save the experiment. Use Serialization (or could use BSON.jl) since DSU.jl is not being updated often.
    serialize(
        joinpath(save_folder_path, "experiment"),
        loaded_experiment,
    )

    # # Simulate and save model.

    preparepath(save_folder_path)
    save_path_SH = joinpath(save_folder_path, "model_SH")
    save_path_cl = joinpath(save_folder_path, "model_CL")
    save_path_SharedT2_fid = joinpath(save_folder_path, "model_SharedT2_FID")
    save_path_CoherenceT2_fid = joinpath(save_folder_path, "model_CoherenceT2_FID")

    # SH and CL surrogate.
    Phys, As, Ws, MSS = setupmodel(
        λ0,
        freq_params,
        molecule_entries,
        H_params_path,
        molecule_mapping_file_path,
        SH_config,
        unique_cs_digits,
        Δc_valid_atol,
        coherence_sum_zero_tol,
        cl_proxy_config,
        save_path_SH,
        save_path_cl,
    )

    # FID surrogate.
    setupproxies(SIG.UseSharedT2(), λ0, As, fid_proxy_config, save_path_SharedT2_fid)
    setupproxies(SIG.UseCoherenceT2(), λ0, As, fid_proxy_config, save_path_CoherenceT2_fid)

    #println("Done.")

    return nothing
end

function preparepath(p::String)
    if !ispath(p)
        mkpath(p)
    end
    return nothing
end

# use reference cs values from H_params_path and molecule_mapping_file_path to get Phys.
function setupmodel(
    λ0::T,
    freq_params::FrequencyParameters{T},
    molecule_entries::Vector{String},
    H_params_path::String,
    molecule_mapping_file_path::String,
    SH_config::HAM.SHConfig{T},
    unique_cs_digits::Int,
    Δc_valid_atol::T,
    coherence_sum_zero_tol::T,
    surrogate_config,
    save_path_SH::String,
    save_path_surrogate::String,
    ) where T <: AbstractFloat

    core_config = SetupConfigCore{T}(
        resonance = SH_config,
        unique_cs_digits = unique_cs_digits,
        Δc_valid_atol = Δc_valid_atol,
        coherence_sum_zero_tol = coherence_sum_zero_tol,
    )
    
    return setupmodel(
        λ0,
        freq_params,
        molecule_entries,
        H_params_path,
        molecule_mapping_file_path,
        SetupConfigCL{T}(
            core = core_config,
            surrogate = surrogate_config,
        ),
        save_path_SH,
        save_path_surrogate,
    )
end

function setupmodel(
    λ0::T,
    freq_params::FrequencyParameters{T},
    molecule_entries::Vector{String},
    H_params_path::String,
    molecule_mapping_file_path::String,
    config::SetupConfig,
    save_path_SH::String,
    save_path_surrogate::String,
    ) where T <: AbstractFloat
    
    Phys = HAM.getphysicalparameters(
        T,
        molecule_entries,
        H_params_path,
        molecule_mapping_file_path;
        unique_cs_digits = config.core.unique_cs_digits,
    )

    As = setupSHmodel(freq_params, molecule_entries, Phys, config.core, save_path_SH)
    Ws, MSS = setupproxies(λ0, As, config.surrogate, save_path_surrogate)

    return Phys, As, Ws, MSS
end

# no error checking on save path.
function setupSHmodel(
    freq_params::FrequencyParameters{T},
    molecule_entries::Vector{String},
    Phys::Vector{HAM.PhysicalParamsType{T}},
    config::SetupConfigCore{T},
    save_path::String,
    ) where T
    
    #@assert ispath(save_path)

    # parse.
    SH_config = config.resonance
    Δc_valid_atol, coherence_sum_zero_tol = config.Δc_valid_atol, config.coherence_sum_zero_tol

    fs, SW, ν_0ppm = freq_params.fs, freq_params.SW, freq_params.ν_0ppm

    # Spin Hamiltonian simulation.
    As, MSPs = HAM.simulate(
        Phys,
        molecule_entries,
        fs,
        SW,
        ν_0ppm,
        SH_config,
    )

    #valid_flag = checkNresonancegroups(As)
    Δ_c_status = checkcoherences(
        As,
        MSPs;
        Δc_valid_atol = Δc_valid_atol,
        coherence_sum_zero_tol = coherence_sum_zero_tol,
    )
    if !Δ_c_status
        println("Warning: Δ_c_status is false.")
    end

    serialize(save_path, (Phys, As, molecule_entries))

    return As
end

function setupproxies(
    λ0::T,
    As::Vector{HAM.SHType{T}},
    config::SIG.CLSurrogateConfig{T},
    save_path::String,
    ) where T <: AbstractFloat

    Bs, MSS, _ = SIG.fitclproxies(As, λ0, config)
    serialize(save_path, (Bs, MSS))

    return Bs, MSS
end

function setupproxies(
    trait::Union{SIG.UseCoherenceT2, SIG.UseSharedT2},
    λ0::T,
    As::Vector{HAM.SHType{T}},
    config::SIG.FIDSurrogateConfig{T},
    save_path::String,
    ) where T <: AbstractFloat

    Cs, MSS, _ = SIG.fitfidproxies(trait, As, λ0, config)
    serialize(save_path, (Cs, MSS))

    return Cs, MSS
end


function checkcoherences(
    As::Vector{HAM.SHType{T}},
    MSPs;
    Δc_valid_atol::T = convert(T, 0.1),
    coherence_sum_zero_tol::T = convert(T, 1e-2),
    ) where T <: AbstractFloat

    AS_Δc_valid, coherence_diagnostics = HAM.checkcoherences(
        As;
        atol = Δc_valid_atol,
    )
    if !AS_Δc_valid
        return false
    end

    # test the partial contributions and full quantum numbers agree.
    for n in eachindex(MSPs)
        for i in eachindex(MSPs[n].spin_systems)
            sp = MSPs[n].spin_systems[i]

            LHS = norm(sum.(sp.partial_quantum_numbers)-sp.quantum_numbers)
            if !(LHS < coherence_sum_zero_tol)
                return false
            end
        end
    end

    return true
end
