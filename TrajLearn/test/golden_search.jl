import Pkg

work_dir = "/home/yizaochen/Projects/traj_learn/TrajLearn"
src_dir = joinpath(work_dir, "src")

Pkg.activate(work_dir)
include(joinpath(src_dir, "FEM.jl"))
include(joinpath(src_dir, "Potential.jl"))
include(joinpath(src_dir, "PhotonOperator.jl"))
include(joinpath(src_dir, "AlphaBeta.jl"))

using .FEM, .Potential, .PhotonOperator, .AlphaBeta
using LinearAlgebra, Printf
using JLD, ArgParse

function get_l(D_guess::Float64, Nv::Int64, rho_s1::Array{Float64,2}, alpha_mat::Array{Float64,2}, Anorm_vec::Array{Float64,2}, big_photon_mat::Array{Float64,3}, idx_array::Vector{Int64}, save_freq::Float64, Lambdas::Vector{Float64}, tau::Int64)
    real_Lambdas = D_guess .* Lambdas
    expLQDT = exp.((-real_Lambdas) .* save_freq) # e^{-HΔt}
    alpha_mat, Anorm_vec = forward(Nv, tau, rho_s1, alpha_mat, Anorm_vec, expLQDT, big_photon_mat, idx_array)
    return -sum(log.(Anorm_vec)) # search maximum -> add negative -> search minimum
end

function bracket(x1::Float64, h::Float64, Nv::Int64, rho_s1::Array{Float64,2}, alpha_mat::Array{Float64,2}, Anorm_vec::Array{Float64,2}, big_photon_mat::Array{Float64,3}, idx_array::Vector{Int64}, save_freq::Float64, Lambdas::Vector{Float64}, tau::Int64)
    c = 1.618033989
    f1 = get_l(x1, Nv, rho_s1, alpha_mat, Anorm_vec, big_photon_mat, idx_array, save_freq, Lambdas, tau)
    x2 = x1 + h # go right to search
    f2 = get_l(x2, Nv, rho_s1, alpha_mat, Anorm_vec, big_photon_mat, idx_array, save_freq, Lambdas, tau)

    if f2 > f1
        h = -h # go to left to search
        x2 = x1 + h
        f2 = get_l(x2, Nv, rho_s1, alpha_mat, Anorm_vec, big_photon_mat, idx_array, save_freq, Lambdas, tau)

        if f2 > f1
            return x2, x1 - h
        end
    end

    # search 10 steps
    for i = 1:50
        h = c * h
        x3 = x2 + h
        f3 = get_l(x3, Nv, rho_s1, alpha_mat, Anorm_vec, big_photon_mat, idx_array, save_freq, Lambdas, tau)
        if f3 > f2
            return x1, x3
        end

        x1 = x2
        x2 = x3
        f1 = f2
        f2 = f3
    end
    println("Bracket did not find a minimum")
end

function search(a::Float64, b::Float64, tol::Float64, Nv::Int64, rho_s1::Array{Float64,2}, alpha_mat::Array{Float64,2}, Anorm_vec::Array{Float64,2}, big_photon_mat::Array{Float64,3}, idx_array::Vector{Int64}, save_freq::Float64, Lambdas::Vector{Float64}, tau::Int64)
    n_interval = ceil(Int64, -2.078087 * log(tol / abs(b-a)))
    R = 0.618033989
    C = 1.0 - R

    # First Telescoping
    x1 = R * a + C * b
    x2 = C * a + R * b
    f1 = get_l(x1, Nv, rho_s1, alpha_mat, Anorm_vec, big_photon_mat, idx_array, save_freq, Lambdas, tau)
    f2 = get_l(x2, Nv, rho_s1, alpha_mat, Anorm_vec, big_photon_mat, idx_array, save_freq, Lambdas, tau)

    # Main Loop
    for i = 1:n_interval
        if f1 > f2
            a = x1
            x1 = x2
            f1 = f2
            x2 = C * a + R * b
            f2 = get_l(x2, Nv, rho_s1, alpha_mat, Anorm_vec, big_photon_mat, idx_array, save_freq, Lambdas, tau)
        else
            b = x2
            x2 = x1
            f2 = f1
            x1 = R * a + C * b
            f1 = get_l(x1, Nv, rho_s1, alpha_mat, Anorm_vec, big_photon_mat, idx_array, save_freq, Lambdas, tau)
        end
    end

    if f1 < f2
        return x1, f1
    else
        return x2, f2
    end
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--host"
            help = "flat_well, harmonic_well, double_well"
            arg_type = String
            default = "double_well"
        "--iter_idx_n"
            help = "iteration index n"
            arg_type = Int
            default = 0
        "--n_frames"
            help = "100000 (high resolution), 10000 (low resolution)"
            arg_type = String
            default = "100000"
        "--sele_iter_id"
            help = "Selected m (the result of learning peq)"
            arg_type = Int
            default = 10
        "--D_guess"
            help = "The guess of diffusion coefficient"
            arg_type = Float64
            default = 40.
    end
    return parse_args(s)
end

function main()
    @show parsed_args = parse_commandline()
    host = parsed_args["host"]
    iter_idx_n = parsed_args["iter_idx_n"]
    n_frames = parsed_args["n_frames"]
    sele_iter_id = parsed_args["sele_iter_id"]
    D_guess = parsed_args["D_guess"]

    #--- Get collocation points and integration kernel for whole domain ---
    Nh = 64    # The number of Spectral element
    Np = 4     # The order of polynomial which used to interpolate and integration
    Nv = 72    # Number of eigenvectors

    # Define Physical Domain
    xratio = 64. # unit: angstrom
    xavg = 0.   # unit: angstrom

    x, w, Ldx, L = getLagrange(Np, xratio/Nh)
    e_norm = x[end] - x[1]
    interpo_xs = x .+ x[end]
    N, xref, w0, Ldx, w = get_fem_xref_weights_basis(Nh, Np, xratio, xavg)
    #-----------------------------------------------------------------------

    #--- Set peq ------------------------------------------------
    data_folder = joinpath(work_dir, "data", host)
    jld_name = @sprintf "peq_update_record_n%s_%s.jld" iter_idx_n n_frames
    f_in = joinpath(data_folder, jld_name)
    xref = load(f_in, "xref")
    p_container = load(f_in, "p_container")
    peq_scan_D = zeros(length(xref), 1)
    peq_scan_D[:,1] = p_container[sele_iter_id+1, :]
    #--------------------------------------------------------------------------

    #--- Do FEM ---------------------------------------------------------------
    println("Start Diagonalization...")
    D_unity = 1e0
    Lambdas, Qx, rho = fem_solve_eigen_by_pref(Nh, Np, xratio, xavg, peq_scan_D, D_unity, Nv)
    println("Complete Diagonalization...")

    if sum(w0 .* Qx[:, 1]) < 0
        Qx = -Qx
    end
    weight_Qx = get_weight_Qx(N, Nv, w0, Qx)
    big_photon_mat = get_big_photon_mat_dirac_delta_analytical(N, Nv, Qx)
    rho_s1 = get_alpha_t0(weight_Qx, rho)
    #--------------------------------------------------------------------------

    #--- Set physical parameters about simulation -----------------------------
    if n_frames == "100000"
        save_freq = 1.0 # unit: T
    else
        save_freq = 10.0
    end
    total_times = 100000 # unit: T
    dt = 2e-5 # unit: s  # Integration Times 1ps
    tau = Int(round(total_times / save_freq, digits=0))  # Number of photons
    n_interval = Int(round(save_freq / dt, digits=0));

    time_info = @sprintf "Save trajectory per %d timesteps. There will be %d data points." n_interval tau
    println(time_info)
    #--------------------------------------------------------------------------

    #--- Read simulated trajectory from hdf5 file -----------------------------
    hdf5_name = @sprintf "traj.%s.frames.hdf5" n_frames
    f_in = joinpath(data_folder, hdf5_name)
    temp = load(f_in, "distance")
    t_record = load(f_in, "time")
    y_record = zeros(tau, 1)
    y_record[:,1] = temp
    idx_array = [find_nearest_point(y_record[time_idx], xref, e_norm, interpo_xs, Np) for time_idx=1:tau]
    #--------------------------------------------------------------------------

    # Initailize Data Storing Matrix
    alpha_mat, beta_mat, Anorm_vec = get_mat_vec(Nv, tau)

    # Golden Search
    x_start = D_guess
    h = 0.1
    x1, x2 = bracket(x_start, h, Nv, rho_s1, alpha_mat, Anorm_vec, big_photon_mat, idx_array, save_freq, Lambdas, tau)
    x, fMin = search(x1, x2, 1e-2, Nv, rho_s1, alpha_mat, Anorm_vec, big_photon_mat, idx_array, save_freq, Lambdas, tau)
    x = x * 10 # convert unit
    @printf "%.5f,%.5f\n" x fMin
end

main()