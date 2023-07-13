module AlphaBeta

    export get_weight_Qx, get_mat_vec, get_loglikelihood_delta, forward_backward_delta, get_alpha_t0, forward

    include("Potential.jl")
    include("FEM.jl")
    include("PhotonOperator.jl")

    using .Potential, .FEM, .PhotonOperator
    using LinearAlgebra, SparseArrays
    #using Optim, LineSearches

    function get_alpha_t0(weight_Qx::Array{Float64,2}, rho_eq::Array{Float64,2})
        return transpose(weight_Qx) * rho_eq
    end

    function get_beta_T(Nv::Int64, weight_Qx::Array{Float64,2})
        beta_T = zeros(Nv,1)
        for idx=1:Nv
            beta_T[idx] = sum(weight_Qx[:, idx])
        end
        return beta_T
    end

    function get_alpha_t0_x_square_norm(alpha_t0::Array{Float64,2}, Qx::Array{Float64,2}, w0::Array{Float64,2})
        alpha_t0_x = Qx * alpha_t0
        alpha_t0_x_square = alpha_t0_x.^2
        alpha_t0_norm = sqrt(sum(w0 .* alpha_t0_x_square))
        return alpha_t0_x, alpha_t0_x_square, alpha_t0_norm
    end

    function get_beta_t_tau(w0, beta_x, Qx, Nv)
        beta = ones(Nv,1)
        temp = w0 .* beta_x
        for idx_eigv in 1:Nv
            beta[idx_eigv] = sum(temp .* Qx[:, idx_eigv])
        end
        return beta
    end

    function gaussian_kde(xref::Array{Float64,2}, y_record::Array{Float64,2}, σ::Float64, w0::Array{Float64,2})
        N = size(xref)[1]
        peq_kde_estimate = zeros(N, 1)
        for μ in y_record
            peq_kde_estimate[:, 1] = peq_kde_estimate[:, 1] .+ gaussian(xref[:, 1], μ, σ)
        end
        peq_kde_estimate[:, 1] = peq_kde_estimate[:, 1] ./ sum(w0 .* peq_kde_estimate[:, 1])
        peq_kde_estimate[:, 1] = max.(peq_kde_estimate[:, 1], 1e-10) 
        return peq_kde_estimate
    end

    function get_D_by_Stokes_Einstein_relation(a::Float64)
        """
        a: Radius of brownian particle, input unit: Å
        """
        kBT = 4.11e-21 # unit: J,   T=298K
        a = a * 1e-10  # Convert from Å to m
        η = 9e-4 # water viscosity, unit: Pa⋅s
        D = kBT / (6π * (η * a)) # unit: m^2/s
        D = D * 1e20 # unit: Å^2/s
        return D
    end

    function initialize(Nh::Int64, Np::Int64, xratio::Float64, xavg::Float64)
        x, w, Ldx, L = getLagrange(Np, xratio/Nh)
        e_norm = x[end] - x[1]
        interpo_xs = x .+ x[end]
        N, xref, w0, Ldx, w = get_fem_xref_weights_basis(Nh, Np, xratio, xavg)
        return e_norm, interpo_xs, xref, w0
    end

    function get_mat_vec(Nv::Int64, number_photon::Int64)
        alpha_mat = zeros(Nv,number_photon)
        beta_mat = zeros(Nv,number_photon)
        Anorm_vec = ones(1,number_photon) # sqrt(c_array)
        return alpha_mat, beta_mat, Anorm_vec
    end

    function get_weight_Qx(N::Int64, Nv::Int64, w0::Array{Float64,2}, Qx::Array{Float64,2})
        weight_Qx = zeros(N, Nv)
        for i = 1:Nv
            weight_Qx[:, i] = w0 .* Qx[:, i]
        end
        return weight_Qx
    end

    function forward(Nv::Int64, number_photon::Int64, rho_s1::Array{Float64,2}, alpha_mat::Array{Float64,2}, Anorm_vec::Array{Float64,2}, expLQDT::Array{Float64,1}, big_photon_mat::Array{Float64,3}, idx_array::Array{Int64,1})
        alpha_hat_prev = zeros(1,Nv)
        alpha_hat_prev[1,:] = rho_s1
        expLQDT = transpose(expLQDT)

        alpha_hat_prev = alpha_hat_prev .* expLQDT # from x0 -> x1

        for time_idx = 1:number_photon 
            psi_photon_psi = big_photon_mat[:,:,idx_array[time_idx]]
        
            alpha_bra = alpha_hat_prev * psi_photon_psi    
            Anorm_vec[time_idx] = abs(alpha_bra[1])
            
            # Normalization
            alpha_hat_bra = alpha_bra ./ Anorm_vec[time_idx]
            alpha_mat[:,time_idx] = alpha_hat_bra
            
            # Time propagation
            alpha_hat_prev = alpha_hat_bra .* expLQDT
        end  
        return alpha_mat, Anorm_vec
    end

    function get_LQ_diff_ij(Nv::Int64, LQ::Array{Float64,1})
        LQ_diff_ij = zeros(Nv,Nv)
        for i in 1:Nv
            LQ_diff_ij[i,:] = 1 ./ (LQ .- LQ[i])
            LQ_diff_ij[i,i] = 0
        end
        return LQ_diff_ij
    end

    function backward(LQ::Array{Float64,1}, dt::Float64, Nv::Int64, number_photon::Int64, beta_mat::Array{Float64,2}, Anorm_vec::Array{Float64,2}, expLQDT::Array{Float64,1}, alpha_mat::Array{Float64,2}, big_photon_mat::Array{Float64,3}, idx_array::Array{Int64,1})
        beta_hat_next = zeros(Nv,1)
        beta_hat_next[1,1] = 1

        LQ_diff_ij = get_LQ_diff_ij(Nv, LQ) # Eq. (63) in JPCB 2013
        someones = ones(1,Nv)
        eLQDT = expLQDT * someones
        exp_ab_mat = zeros(Nv,Nv)

        for time_idx = number_photon:-1:1
            #beta_mat[:,time_idx] = beta_hat_next[:,1] ### OLD
            psi_photon_psi = big_photon_mat[:,:,idx_array[time_idx]]

            # Photon operation and Normalization
            beta_hat_next = (psi_photon_psi * beta_hat_next) ./ Anorm_vec[time_idx]
            beta_mat[:,time_idx] = beta_hat_next[:,1]

            # Eq. (64) and Eq. (63)
            if time_idx != 1
                outer= alpha_mat[:, time_idx-1] * beta_hat_next'
                exp_ab_mat = exp_ab_mat .+ outer .* ( diagm(expLQDT * dt) + LQ_diff_ij .* (eLQDT-eLQDT'))
            end

            # Time propagation
            beta_hat_next = expLQDT .* beta_hat_next
        end
        return exp_ab_mat, beta_mat
    end

    function get_exp_ab_mat(Lambdas::Array{Float64,1}, Nv::Int64, tau::Int64, alpha_mat::Array{Float64,2}, beta_mat::Array{Float64,2}, dt::Float64)
        LQ_diff_ij = get_LQ_diff_ij(Nv, Lambdas) # Eq. (63) in JPCB 2013
        expLQDT = exp.(-Lambdas .* dt)
        someones = ones(1,Nv)
        eLQDT = expLQDT * someones

        exp_ab_mat = zeros(Nv,Nv)
        for idx in 1:tau
            outer= alpha_mat[:, idx] * beta_mat[:, idx]'
            exp_ab_mat = exp_ab_mat .+ outer .* ( diagm(expLQDT * dt) + LQ_diff_ij .* (eLQDT-eLQDT')) # Eq. (64)
        end
        return exp_ab_mat
    end


    function forward_backward(Nh::Int64, Np::Int64, xratio::Float64, xavg::Float64, peq::Array{Float64,2}, D_guess::Array{Float64,1}, Nv::Int64, tau::Int64, y_record::Array{Float64,2}, dt::Float64, k_photon::Float64)
        e_norm, interpo_xs, xref, w0 = initialize(Nh, Np, xratio, xavg)
        D_unity = 1e0
        Lambdas, Qx, rho = fem_solve_eigen_by_pref(Nh, Np, xratio, xavg, peq, D_unity, Nv)
        real_Lambdas = D_guess .* Lambdas
        N  = Nh*Np - Nh + 1 # Total number of nodes
        weight_Qx = get_weight_Qx(N, Nv, w0, Qx)
        alpha_mat, beta_mat, Anorm_vec = get_mat_vec(Nv, tau) # Initailize Data Storing Matrix
        expLQDT = exp.((-real_Lambdas) .* dt) # e^{-HΔt}

        big_photon_mat = get_big_photon_mat(N, Nv, w0, k_photon, xref, Qx) # Carteisian Space Photon Operator
        idx_array = [find_nearest_point(y_record[time_idx], xref, e_norm, interpo_xs, Np) for time_idx=1:tau]

        # Forward Algorithm
        rho_s1 = get_alpha_t0(weight_Qx, rho)
        alpha_mat, Anorm_vec = forward(Nv, tau, rho_s1, alpha_mat, Anorm_vec, expLQDT, big_photon_mat, idx_array)

        # Backward Algorithm
        exp_ab_mat, beta_mat = backward(real_Lambdas, dt, Nv, tau, beta_mat, Anorm_vec, expLQDT, alpha_mat, big_photon_mat, idx_array)

        # Get log-likelihood
        log_likelihood = sum(log.(Anorm_vec)) # Eq. (41)

        # Eq. (72) and Eq. (78)
        peq_new = diag(Qx * exp_ab_mat * Qx')
        peq_new_normalize = peq_new ./ sum(w0 .* peq_new)
        return peq_new_normalize, log_likelihood
    end


    function forward_backward_dump(Nh::Int64, Np::Int64, xratio::Float64, xavg::Float64, peq::Array{Float64,2}, D_guess::Array{Float64,1}, Nv::Int64, tau::Int64, y_record::Array{Float64,2}, dt::Float64, k_photon::Float64)
        e_norm, interpo_xs, xref, w0 = initialize(Nh, Np, xratio, xavg)
        D_unity = 1e0
        Lambdas, Qx, rho = fem_solve_eigen_by_pref(Nh, Np, xratio, xavg, peq, D_unity, Nv)
        real_Lambdas = D_guess .* Lambdas
        N  = Nh*Np - Nh + 1 # Total number of nodes
        weight_Qx = get_weight_Qx(N, Nv, w0, Qx)
        alpha_mat, beta_mat, Anorm_vec = get_mat_vec(Nv, tau) # Initailize Data Storing Matrix
        expLQDT = exp.((-real_Lambdas) .* dt) # e^{-HΔt}

        big_photon_mat = get_big_photon_mat(N, Nv, w0, k_photon, xref, Qx) # Carteisian Space Photon Operator
        idx_array = [find_nearest_point(y_record[time_idx], xref, e_norm, interpo_xs, Np) for time_idx=1:tau]

        # Forward Algorithm
        rho_s1 = get_alpha_t0(weight_Qx, rho)
        alpha_mat, Anorm_vec = forward(Nv, tau, rho_s1, alpha_mat, Anorm_vec, expLQDT, big_photon_mat, idx_array)

        # Backward Algorithm
        exp_ab_mat, beta_mat = backward(real_Lambdas, dt, Nv, tau, beta_mat, Anorm_vec, expLQDT, alpha_mat, big_photon_mat, idx_array)

        # Get log-likelihood
        log_likelihood = sum(log.(Anorm_vec)) # Eq. (41)

        # Eq. (72) and Eq. (78)
        peq_new = diag(Qx * exp_ab_mat * Qx')
        peq_new_normalize = peq_new ./ sum(w0 .* peq_new)
        return peq_new_normalize, log_likelihood, alpha_mat, Anorm_vec, exp_ab_mat, beta_mat, expLQDT, Lambdas, Qx
    end

    function forward_backward_delta(Nh::Int64, Np::Int64, xratio::Float64, xavg::Float64, peq::Array{Float64,2}, D_guess::Array{Float64,1}, Nv::Int64, tau::Int64, y_record::Array{Float64,2}, dt::Float64)
        e_norm, interpo_xs, xref, w0 = initialize(Nh, Np, xratio, xavg)
        D_unity = 1e0
        Lambdas, Qx, rho = fem_solve_eigen_by_pref(Nh, Np, xratio, xavg, peq, D_unity, Nv)
        real_Lambdas = D_guess .* Lambdas
        N  = Nh*Np - Nh + 1 # Total number of nodes
        weight_Qx = get_weight_Qx(N, Nv, w0, Qx)
        alpha_mat, beta_mat, Anorm_vec = get_mat_vec(Nv, tau) # Initailize Data Storing Matrix
        expLQDT = exp.((-real_Lambdas) .* dt) # e^{-HΔt}

        big_photon_mat = get_big_photon_mat_dirac_delta_analytical(N, Nv, Qx)
        idx_array = [find_nearest_point(y_record[time_idx], xref, e_norm, interpo_xs, Np) for time_idx=1:tau]

        # Forward Algorithm
        rho_s1 = get_alpha_t0(weight_Qx, rho)
        alpha_mat, Anorm_vec = forward(Nv, tau, rho_s1, alpha_mat, Anorm_vec, expLQDT, big_photon_mat, idx_array)

        # Backward Algorithm
        exp_ab_mat, beta_mat = backward(real_Lambdas, dt, Nv, tau, beta_mat, Anorm_vec, expLQDT, alpha_mat, big_photon_mat, idx_array)

        # Get log-likelihood
        log_likelihood = sum(log.(Anorm_vec)) # Eq. (41)

        # Eq. (72) and Eq. (78)
        peq_new = diag(Qx * exp_ab_mat * Qx')
        peq_new_normalize = peq_new ./ sum(w0 .* peq_new)
        return peq_new_normalize, log_likelihood
    end

    function forward_backward_delta_damping(Nh::Int64, Np::Int64, xratio::Float64, xavg::Float64, peq::Array{Float64,2}, D_guess::Array{Float64,1}, Nv::Int64, tau::Int64, y_record::Array{Float64,2}, dt::Float64, t_D::Float64)
        e_norm, interpo_xs, xref, w0 = initialize(Nh, Np, xratio, xavg)
        D_unity = 1e0
        Lambdas, Qx, rho = fem_solve_eigen_by_pref(Nh, Np, xratio, xavg, peq, D_unity, Nv)
        real_Lambdas = D_guess .* Lambdas
        N  = Nh*Np - Nh + 1 # Total number of nodes
        weight_Qx = get_weight_Qx(N, Nv, w0, Qx)
        alpha_mat, beta_mat, Anorm_vec = get_mat_vec(Nv, tau) # Initailize Data Storing Matrix
        expLQDT = exp.((-real_Lambdas) .* dt) # e^{-HΔt}

        big_photon_mat = get_big_photon_mat_dirac_delta_damping(N, Nv, Qx, t_D, Lambdas)
        idx_array = [find_nearest_point(y_record[time_idx], xref, e_norm, interpo_xs, Np) for time_idx=1:tau]

        # Forward Algorithm
        rho_s1 = get_alpha_t0(weight_Qx, rho)
        alpha_mat, Anorm_vec = forward(Nv, tau, rho_s1, alpha_mat, Anorm_vec, expLQDT, big_photon_mat, idx_array)

        # Backward Algorithm
        exp_ab_mat, beta_mat = backward(real_Lambdas, dt, Nv, tau, beta_mat, Anorm_vec, expLQDT, alpha_mat, big_photon_mat, idx_array)

        # Get log-likelihood
        log_likelihood = sum(log.(Anorm_vec)) # Eq. (41)

        # Eq. (72) and Eq. (78)
        peq_new = diag(Qx * exp_ab_mat * Qx')
        peq_new_normalize = peq_new ./ sum(w0 .* peq_new)
        return peq_new_normalize, log_likelihood
    end

    function get_loglikelihood(Nh::Int64, Np::Int64, xratio::Float64, xavg::Float64, peq::Array{Float64,2}, D_guess::Array{Float64,1}, Nv::Int64, tau::Int64, y_record::Array{Float64,2}, dt::Float64, k_photon::Float64)
        e_norm, interpo_xs, xref, w0 = initialize(Nh, Np, xratio, xavg)

        D_unity = 1e0
        Lambdas, Qx, rho = fem_solve_eigen_by_pref(Nh, Np, xratio, xavg, peq, D_unity, Nv)
        real_Lambdas = D_guess .* Lambdas

        N  = Nh*Np - Nh + 1 # Total number of nodes
        weight_Qx = get_weight_Qx(N, Nv, w0, Qx)
        alpha_mat, beta_mat, Anorm_vec = get_mat_vec(Nv, tau) # Initailize Data Storing Matrix
        expLQDT = exp.((-real_Lambdas) .* dt) # e^{-HΔt}

        big_photon_mat = get_big_photon_mat(N, Nv, w0, k_photon, xref, Qx) # Carteisian Space Photon Operator
        idx_array = [find_nearest_point(y_record[time_idx], xref, e_norm, interpo_xs, Np) for time_idx=1:tau]

        # Forward Algorithm
        rho_s1 = get_alpha_t0(weight_Qx, rho)
        alpha_mat, Anorm_vec = forward(Nv, tau, rho_s1, alpha_mat, Anorm_vec, expLQDT, big_photon_mat, idx_array)
        return sum(log.(Anorm_vec)) # Eq. (41)
    end

    function get_loglikelihood_delta(Nh::Int64, Np::Int64, xratio::Float64, xavg::Float64, peq::Array{Float64,2}, D_guess::Array{Float64,1}, Nv::Int64, tau::Int64, y_record::Array{Float64,2}, dt::Float64)
        e_norm, interpo_xs, xref, w0 = initialize(Nh, Np, xratio, xavg)

        D_unity = 1e0
        Lambdas, Qx, rho = fem_solve_eigen_by_pref(Nh, Np, xratio, xavg, peq, D_unity, Nv)
        real_Lambdas = D_guess .* Lambdas

        N  = Nh*Np - Nh + 1 # Total number of nodes
        weight_Qx = get_weight_Qx(N, Nv, w0, Qx)
        alpha_mat, beta_mat, Anorm_vec = get_mat_vec(Nv, tau) # Initailize Data Storing Matrix
        expLQDT = exp.((-real_Lambdas) .* dt) # e^{-HΔt}

        big_photon_mat = get_big_photon_mat_dirac_delta_analytical(N, Nv, Qx)
        idx_array = [find_nearest_point(y_record[time_idx], xref, e_norm, interpo_xs, Np) for time_idx=1:tau]

        # Forward Algorithm
        rho_s1 = get_alpha_t0(weight_Qx, rho)
        alpha_mat, Anorm_vec = forward(Nv, tau, rho_s1, alpha_mat, Anorm_vec, expLQDT, big_photon_mat, idx_array)
        return sum(log.(Anorm_vec)) # Eq. (41)
    end
end