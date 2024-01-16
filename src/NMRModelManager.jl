module NMRModelManager

using LinearAlgebra
using FFTW
using Serialization

using SparseArrays
import OSQP

import NMRDataSetup
const DSU = NMRDataSetup

import NMRSpecifyRegions

import NMRSignalSimulator
const SIG = NMRSignalSimulator
const NMRHamiltonian = NMRSignalSimulator.NMRHamiltonian
const HAM = NMRHamiltonian
const JSON3 = HAM.JSON3

import BSON

const MoleculeType = NMRSignalSimulator.MoleculeType
const SpinSysParams = NMRSignalSimulator.SpinSysParams
const CoherenceShift = NMRSignalSimulator.CoherenceShift
const CoherencePhase = NMRSignalSimulator.CoherencePhase
const SharedT2 = NMRSignalSimulator.SharedT2
const SHType = NMRHamiltonian.SHType
const PhysicalParamsType = NMRHamiltonian.PhysicalParamsType
const CLMixtureSpinSys = NMRSignalSimulator.CLMixtureSpinSys
const FIDMixtureSpinSys = NMRSignalSimulator.FIDMixtureSpinSys

# constant values.
function twopi(::Type{Float32})::Float32
    return 6.2831855f0 #convert(T, 2*π)
end

function twopi(::Type{Float64})::Float64
    return 6.283185307179586 #convert(T, 2*π)
end

function twopi(::Type{T})::T where T <: AbstractFloat
    return convert(T, 2*π)
end

include("types.jl")
include("checks.jl")

include("./objective/batch/b_models.jl")
include("./objective/cost.jl")
include("./objective/batch/b_derivatives.jl")

# IO of model.
include("./IO/load_experiments.jl")
include("./IO/IO.jl")


# fitting the model.
include("./fit/var_utils.jl") # these should be exported. include types.
include("./fit/BLS.jl") # sets the basis for the model fit cost function.
include("./fit/costfunc.jl")

include("./fit/intervals.jl")

include("frontend.jl")
include("analysis.jl")

export resetparameter!,

# model fit objective
# GradientTrait,
# UseGradient,
# IgnoreGradient,

costfuncsetup,
evalcost,
evalcost!,
# updategradientbuffer!,
constructdesignmatrix!,

CostFuncBuffer,
CostFuncBuffer,
BatchEvalBuffer
#packagedevalgradient!

resetmodel!,
getdefaultparameter

end # module NMRModelManager
