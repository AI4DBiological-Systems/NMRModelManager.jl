
##### datatypes related to batch evaluation of model.

"""
```abstract type GradientTrait end```
"""
abstract type GradientTrait end

"""
```
struct UseGradient <: GradientTrait end
```
"""
struct UseGradient <: GradientTrait end

"""
```
struct IgnoreGradient <: GradientTrait end
```
"""
struct IgnoreGradient <: GradientTrait end

"""
```
struct BatchEvalBuffer{T}
    U_rad::Vector{T}
    
    # buffers.
    A::Matrix{Complex{T}} # length(U_rad) x getNentries(model_params)    
    r::Vector{Vector{Vector{Vector{T}}}} # getNgroups(As), length(U_rad)
    re_evals::Vector{Vector{Vector{Vector{T}}}} # getNgroups(As), length(U_rad)
    im_evals::Vector{Vector{Vector{Vector{T}}}} # getNgroups(As), length(U_rad)

    grad_re_evals::Vector{Vector{Vector{Vector{Vector{T}}}}} # indices: entry, spin sys, group, m-th U_rad position, 1 for dq_dr or 2 for dq_dλ.
    grad_im_evals::Vector{Vector{Vector{Vector{Vector{T}}}}}
end
```
"""
struct BatchEvalBuffer{T}
    U_rad::Vector{T}
    
    # buffers.
    A::Matrix{Complex{T}} # length(U_rad) x getNentries(model_params)    
    r::Vector{Vector{Vector{Vector{T}}}} # getNgroups(As), length(U_rad)
    re_evals::Vector{Vector{Vector{Vector{T}}}} # getNgroups(As), length(U_rad)
    im_evals::Vector{Vector{Vector{Vector{T}}}} # getNgroups(As), length(U_rad)

    grad_re_evals::Vector{Vector{Vector{Vector{Vector{T}}}}} # indices: entry, spin sys, group, m-th U_rad position, 1 for dq_dr or 2 for dq_dλ.
    grad_im_evals::Vector{Vector{Vector{Vector{Vector{T}}}}}
end

"""
```
function BatchEvalBuffer(
    U_rad::Vector{T},
    MSS::CLMixtureSpinSys,
)::BatchEvalBuffer{T} where T <: AbstractFloat
```

Convenience constructor for type `BatchEvalBuffer`.
"""
function BatchEvalBuffer(
    U_rad::Vector{T},
    MSS::CLMixtureSpinSys,
    )::BatchEvalBuffer{T} where T <: AbstractFloat

    M = length(U_rad)

    A = Matrix{Complex{T}}(undef, M, SIG.getNentries(MSS))

    r = createresonancegroupbuffer(T, MSS.srs, M)
    re_evals = createresonancegroupbuffer(T, MSS.srs, M)
    im_evals = createresonancegroupbuffer(T, MSS.srs, M)

    grad_re_evals = createresonancegroupbuffer(T, MSS.srs, M, 2)
    grad_im_evals = createresonancegroupbuffer(T, MSS.srs, M, 2)

    return BatchEvalBuffer(U_rad, A, r, re_evals, im_evals, grad_re_evals, grad_im_evals)
end

function createresonancegroupbuffer(
    ::Type{T},
    srs,
    M::Int,
    )::Vector{Vector{Vector{Vector{T}}}} where T <: AbstractFloat

    re_evals = Vector{Vector{Vector{Vector{T}}}}(undef, length(srs))
    
    for n in eachindex(srs)
        re_evals[n] = Vector{Vector{Vector{T}}}(undef, length(srs[n]))

        for i in eachindex(srs[n])
            re_evals[n][i] = Vector{Vector{T}}(undef, length(srs[n][i]))

            for k in eachindex(srs[n][i])
                re_evals[n][i][k] = Vector{T}(undef, M)
            end
        end
    end

    return re_evals
end

function createresonancegroupbuffer(
    ::Type{T},
    srs,
    M::Int,
    D::Int,
    )::Vector{Vector{Vector{Vector{Vector{T}}}}} where T <: AbstractFloat

    grad_re_evals = Vector{Vector{Vector{Vector{Vector{T}}}}}(undef, length(srs))
    
    for n in eachindex(srs)
        grad_re_evals[n] = Vector{Vector{Vector{Vector{T}}}}(undef, length(srs[n]))

        for i in eachindex(srs[n])
            grad_re_evals[n][i] = Vector{Vector{Vector{T}}}(undef, length(srs[n][i]))

            for k in eachindex(srs[n][i])

                grad_re_evals[n][i][k] = collect( Vector{T}(undef, D) for m = 1:M )    
            end
        end
    end

    return grad_re_evals
end

###### methods.

"""
```
function updatebuffer!(B::BatchEvalBuffer, MSS::CLMixtureSpinSys, ::SIG.AllVars)
```

Update all variables of `B` with the contents of MSS.
"""
function updatebuffer!(B::BatchEvalBuffer, MSS, ::SIG.AllVars)
    return updatebuffer!(B, MSS)
end

"""
```
function updatebuffer!(B::BatchEvalBuffer, MSS::CLMixtureSpinSys)
```

Update all variables of `B` with the contents of MSS.
"""
function updatebuffer!(
    B::BatchEvalBuffer{T},
    MSS,
    ) where T

    srs, sis = MSS.srs, MSS.sis
    shifts, T2s = MSS.shifts, MSS.T2s
    U_rad, r, re_evals, im_evals = B.U_rad, B.r, B.re_evals, B.im_evals

    for n in eachindex(r)
        for i in eachindex(r[n])
            for k in eachindex(r[n][i])

                λ = SIG.fetchT2(T2s, n, i, k)

                # r = ω - ζ
                ζ = SIG.fetchshift(shifts, n, i, k)
                
                r[n][i][k][:] = U_rad .- ζ
                re_evals[n][i][k][:] = srs[n][i][k].(r[n][i][k], λ)
                im_evals[n][i][k][:] = sis[n][i][k].(r[n][i][k], λ)
            end
        end
    end

    return nothing
end

"""
```
function updatebuffer!(
    B::BatchEvalBuffer{T},
    MSS::CLMixtureSpinSys,
    active_systems::Vector{Tuple{Int,Int}},
    ) where T
```

Updates the containers in `B` with the parameter-related quantities in `MSS`.
Only the parameters belonging in the spin systems that are represented in `active_systems` are updated.
If `active_systems[k] == (n,i)`, then it means the n-th compound entry and i-th spin system is to be updated.
"""
function updatebuffer!(
    B::BatchEvalBuffer{T},
    MSS::CLMixtureSpinSys,
    active_systems::Vector{Tuple{Int,Int}},
    ) where T
    # this is the version used for select variable updates.
    # the contents insie the loop is exactly the same as updatebuffer!() without `active_systems`.

    srs, sis = MSS.srs, MSS.sis
    shifts, T2s = MSS.shifts, MSS.T2s
    U_rad, r, re_evals, im_evals = B.U_rad, B.r, B.re_evals, B.im_evals

    for (n,i) in active_systems
        for k in eachindex(r[n][i])

            λ = SIG.fetchT2(T2s, n, i, k)

            # r = ω - ζ
            ζ = SIG.fetchshift(shifts, n, i, k)
            
            r[n][i][k][:] = U_rad .- ζ
            re_evals[n][i][k][:] = srs[n][i][k].(r[n][i][k], λ)
            im_evals[n][i][k][:] = sis[n][i][k].(r[n][i][k], λ)
        end
    end

    return nothing
end


"""
```
function updategradientbuffer!(
    B::BatchEvalBuffer{T},
    MSS::CLMixtureSpinSys,
) where T
```

Updates the containers in `B` with the parameter-related quantities in `MSS`.
"""
function updategradientbuffer!(
    B::BatchEvalBuffer{T},
    MSS::CLMixtureSpinSys{T,SST},
    ) where {T, SST}

    # not using multiple dispatch with traits in inner-most loop because it might cause performance hit.

    srs, sis, ∇srs!, ∇sis! = MSS.srs, MSS.sis, MSS.∇srs!, MSS.∇sis!
    shifts, T2s = MSS.shifts, MSS.T2s
    U_rad, r, re_evals, im_evals = B.U_rad, B.r, B.re_evals, B.im_evals
    grad_re_evals, grad_im_evals = B.grad_re_evals, B.grad_im_evals

    for n in eachindex(r)
        for i in eachindex(r[n])

            re_ni = re_evals[n][i]
            im_ni = im_evals[n][i]
            srs_ni = srs[n][i]
            sis_ni = sis[n][i]
            r_ni = r[n][i]
            gre_ni = grad_re_evals[n][i]
            gim_ni = grad_im_evals[n][i]
            gsrs_ni = ∇srs![n][i]
            gsis_ni = ∇sis![n][i]

            for k in eachindex(r[n][i])

                λ = SIG.fetchT2(T2s, n, i, k)

                # r = ω - ζ
                ζ = SIG.fetchshift(shifts, n, i, k)

                updaterad!(r_ni[k], U_rad,  ζ)
                itpquery!(re_ni[k], srs_ni[k], r_ni[k], λ)
                itpquery!(im_ni[k], sis_ni[k], r_ni[k], λ)

                updategradientbuffer!(
                    gre_ni[k],
                    gim_ni[k],
                    gsrs_ni[k],
                    gsis_ni[k],
                    r_ni[k],
                    λ,
                )

            end
        end
    end

    return nothing
end

function itpquery!(out::Vector{T}, f, r::Vector{T}, λ::T)::Nothing where T

    @assert length(out) == length(r)

    for j in eachindex(out)
        out[j] = f(r[j], λ)
    end

    return nothing
end

function updaterad!(r::Vector{T}, U_rad, ζ::T) where T <: AbstractFloat
    
    @assert length(r) == length(U_rad)

    for m in eachindex(U_rad)
        r[m] = U_rad[m] - ζ
    end

    return nothing
end

"""
```
function updategradientbuffer!(
    grad_re_evals::Vector{Vector{T}},
    grad_im_evals::Vector{Vector{T}},
    ∇sr!,
    ∇si!,
    r::Vector{T},
    λ::T,
) where T <: AbstractFloat
```

Updates the containers `grad_re_evals` and `grad_im_evals` the other inputs.
"""
function updategradientbuffer!(
    grad_re_evals::Vector{Vector{T}},
    grad_im_evals::Vector{Vector{T}},
    ∇sr!,
    ∇si!,
    r::Vector{T},
    λ::T,
    ) where T <: AbstractFloat

    @assert length(r) == length(grad_re_evals) == length(grad_im_evals)

    for m in eachindex(r)
        ∇sr!(grad_re_evals[m], r[m], λ)
        ∇si!(grad_im_evals[m], r[m], λ)

        # if !isallfinite(grad_re_evals[m])
        #     @show m, r[m], λ, grad_re_evals[m]
        # end
        # @assert isallfinite(grad_re_evals[m])
        # @assert isallfinite(grad_im_evals[m])
    end

    return nothing
end

"""
```
function updategradientbuffer!(B::BatchEvalBuffer, MSS, ::SIG.AllVars)
```

perform updategradientbuffer!(B::BatchEvalBuffer, MSS) for all parameters.
"""
function updategradientbuffer!(B::BatchEvalBuffer, MSS, ::SIG.AllVars)
    return updategradientbuffer!(B, MSS)
end

"""
```
function updategradientbuffer!(
    B::BatchEvalBuffer{T},
    MSS::CLMixtureSpinSys,
    active_systems::Vector{Tuple{Int,Int}},
) where T
```

Updates the containers in `B` with the parameter-related quantities in `MSS`.
Only the parameters belonging in the spin systems that are represented in `active_systems` are updated.
If `active_systems[k] == (n,i)`, then it means the n-th compound entry and i-th spin system is to be updated.

"""
function updategradientbuffer!(
    B::BatchEvalBuffer{T},
    MSS::CLMixtureSpinSys,
    active_systems::Vector{Tuple{Int,Int}},
    ) where T

    # this is the version used for select variable updates.
    # the contents insie the loop is exactly the same as updategradientbuffer!() without `active_systems`.

    srs, sis, ∇srs!, ∇sis! = MSS.srs, MSS.sis, MSS.∇srs!, MSS.∇sis!
    shifts, T2s = MSS.shifts, MSS.T2s
    U_rad, r, re_evals, im_evals = B.U_rad, B.r, B.re_evals, B.im_evals
    grad_re_evals, grad_im_evals = B.grad_re_evals, B.grad_im_evals

    for (n,i) in active_systems
        re_ni = re_evals[n][i]
        im_ni = im_evals[n][i]
        srs_ni = srs[n][i]
        sis_ni = sis[n][i]
        r_ni = r[n][i]
        gre_ni = grad_re_evals[n][i]
        gim_ni = grad_im_evals[n][i]
        gsrs_ni = ∇srs![n][i]
        gsis_ni = ∇sis![n][i]

        for k in eachindex(r[n][i])
            λ = SIG.fetchT2(T2s, n, i, k)

            # r = ω - ζ
            ζ = SIG.fetchshift(shifts, n, i, k)
            
            # r[n][i][k][:] = U_rad .- ζ
            # re_evals[n][i][k][:] = srs[n][i][k].(r[n][i][k], λ)
            # im_evals[n][i][k][:] = sis[n][i][k].(r[n][i][k], λ)

            # updaterad!(r[n][i][k], U_rad,  ζ)
            # itpquery!(re_evals[n][i][k], srs[n][i][k], r[n][i][k], λ)
            # itpquery!(im_evals[n][i][k], sis[n][i][k], r[n][i][k], λ)

            updaterad!(r_ni[k], U_rad,  ζ)
            itpquery!(re_ni[k], srs_ni[k], r_ni[k], λ)
            itpquery!(im_ni[k], sis_ni[k], r_ni[k], λ)

            # timing experiments.

            # updategradientbuffer!(
            #     grad_re_evals[n][i][k],
            #     grad_im_evals[n][i][k],
            #     ∇srs![n][i][k],
            #     ∇sis![n][i][k],
            #     r[n][i][k],
            #     λ,
            # )

            updategradientbuffer!(
                gre_ni[k],
                gim_ni[k],
                gsrs_ni[k],
                gsis_ni[k],
                r_ni[k],
                λ,
            )
        end
    end

    return nothing
end


########## design matrix construction.

DOCSTRING_design_matrix = """
Each row corresponds to a frequency positions in `U_rad`, and each column corresponds to  a molecule entry. The returned variable `B` is a reinterpreted version of `A`, which is a real-valued matrix where each row in `A` is separated into consecutive rows in `B`.
"""

# for verification.
"""
```
constructdesignmatrix!(
    A::Matrix{Complex{T}},
    MSS::CLMixtureSpinSys,
    U_rad,
)::Matrix{T} where T <: AbstractFloat
```

Updates the function evaluations buffers `X.re_evals` and `X.im_evals` from `MSS`, then update the complex-valued design matrix `A`. The supplied `U_rad` is the frequency positions used for the evaluation.
    
$DOCSTRING_design_matrix
"""
function constructdesignmatrix!( # check if unused. incorrect due to evalsystems not computing phase?
    A::Matrix{Complex{T}},
    MSS::CLMixtureSpinSys,
    U_rad,
    )::Matrix{T} where T <: AbstractFloat
    # assuming all arrays are 1-indexing and stride 1.

    N = SIG.getNentries(MSS)
    @assert size(A) == (length(U_rad), N)

    for n in axes(A,2)
        for m in axes(A,1)

            A[m,n] = evalsystems(
                U_rad[m],
                MSS.srs[n],
                MSS.sis[n],
                MSS.shifts[n],
                MSS.phases[n],
                MSS.T2s[n],
            )
        end
    end
    
    B = reinterpret(T, A)
    return B
end

# in actual usage, we separating surrogate evaluations & gradient with the costfunction evaluation and gradient. This is for speed.
"""
constructdesignmatrix!(
    ::UseGradient,
    X::BatchEvalBuffer{T},
    MSS::CLMixtureSpinSys,
    )::Matrix{T} where T <: AbstractFloat

Updates the function evaluations buffers `X.re_evals` and `X.im_evals` from `MSS`. Updates the gradient evaluations buffers `X.grad_re_evals` and `X.grad_im_evals` from `MSS`.  Then, compute the complex-valued design matrix `X.A`.

Let `A` be `X.A` and `U_rad` be `X.U_rad`:
$DOCSTRING_design_matrix
"""
function constructdesignmatrix!(
    ::UseGradient,
    X::BatchEvalBuffer{T},
    MSS::CLMixtureSpinSys,
    )::Matrix{T} where T <: AbstractFloat
    # assuming all arrays are 1-indexing and stride 1.

    updategradientbuffer!(X, MSS)
    return constructdesignmatrix!(X, MSS)
end

"""
```
constructdesignmatrix!(
    ::IgnoreGradient,
    X::BatchEvalBuffer{T},
    MSS::CLMixtureSpinSys,
    )::Matrix{T} where T <: AbstractFloat
```

Updates the function evaluations buffers `X.re_evals` and `X.im_evals` from `MSS`, then compute the complex-valued design matrix `X.A`.
    
Let `A` be `X.A` and `U_rad` be `X.U_rad`:
$DOCSTRING_design_matrix
"""
function constructdesignmatrix!(
    ::IgnoreGradient,
    X::BatchEvalBuffer{T},
    MSS::CLMixtureSpinSys,
    )::Matrix{T} where T <: AbstractFloat
    # assuming all arrays are 1-indexing and stride 1.

    updatebuffer!(X, MSS)
    return constructdesignmatrix!(X, MSS)
end

"""
constructdesignmatrix!(
    X::BatchEvalBuffer{T},
    MSS::CLMixtureSpinSys,
    )::Matrix{T} where T <: AbstractFloat

Assumes the function evaluation and gradient buffers `X.re_evals` and `X.im_evals` is already updated from `MSS`, and does not update it.
"""
function constructdesignmatrix!(
    X::BatchEvalBuffer{T},
    MSS::CLMixtureSpinSys,
    )::Matrix{T} where T <: AbstractFloat

    phases = MSS.phases
    A, re_evals, im_evals = X.A, X.re_evals, X.im_evals

    N = SIG.getNentries(MSS)
    @assert size(A) == (length(X.U_rad), N)
    @assert length(re_evals) == length(im_evals) == N

    fill!(A, zero(Complex{T}))

    for n in axes(A,2)

        for i in eachindex(X.re_evals[n])
            for k in eachindex(X.re_evals[n][i])
                
                for m in axes(A,1)

                    A[m,n] += Complex(
                        re_evals[n][i][k][m],
                        im_evals[n][i][k][m],
                    )*cis(phases[n].β[i][k])
                end
            end
        end
    end
    
    B = reinterpret(T, A)
    return B
end

"""
```
function constructdesignmatrix!(
    X::BatchEvalBuffer{T},
    MSS::CLMixtureSpinSys,
    active_systems::Vector{Tuple{Int,Int}},
)::Matrix{T} where T <: AbstractFloat
```

Assumes the function evaluation and gradient buffers `X.re_evals` and `X.im_evals` is already updated from `MSS`, and does not update it.
Only the parameters belonging in the spin systems that are represented in `active_systems` are updated.
If `active_systems[k] == (n,i)`, then it means the n-th compound entry and i-th spin system is to be updated.

"""
function constructdesignmatrix!(
    X::BatchEvalBuffer{T},
    MSS::CLMixtureSpinSys,
    active_systems::Vector{Tuple{Int,Int}},
    )::Matrix{T} where T <: AbstractFloat

    phases = MSS.phases
    A, re_evals, im_evals = X.A, X.re_evals, X.im_evals

    N = SIG.getNentries(MSS)
    @assert size(A) == (length(X.U_rad), N)
    @assert length(re_evals) == length(im_evals) == N

    #fill!(A, zero(Complex{T}))
    for (n,i) in active_systems
        for m in axes(A,1)
            A[m,n] = zero(Complex{T})
        end
    end

    for (n,i) in active_systems
        for k in eachindex(X.re_evals[n][i])
            
            for m in axes(A,1)

                A[m,n] += Complex(
                    re_evals[n][i][k][m],
                    im_evals[n][i][k][m],
                )*cis(phases[n].β[i][k])
            end
        end
    end
    
    B = reinterpret(T, A)
    return B
end