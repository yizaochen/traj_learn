import Pkg

work_dir = "/home/yizaochen/Projects/traj_learn/TrajLearn"
src_dir = joinpath(work_dir, "src")

Pkg.activate(work_dir)
include(joinpath(src_dir, "FEM.jl"))
include(joinpath(src_dir, "Potential.jl"))
include(joinpath(src_dir, "PhotonOperator.jl"))
include(joinpath(src_dir, "AlphaBeta.jl"))

using .FEM, .Potential, .PhotonOperator, .AlphaBeta
using Printf
using JLD, ArgParse, SavitzkyGolay

function potential_uniform(x)
    return 1.0
end

function expectation_maximization(max_n_iteration::Int64, N::Int64, Nh::Int64, Np::Int64, xratio::Float64, xavg::Float64, Nv::Int64, tau::Int64, y_record::Array{Float64,2}, save_freq::Float64, p0::Array{Float64,2}, D_guess_array::Array{Float64,1}, w0::Array{Float64,2})
    p_container = zeros(Float64, max_n_iteration+1, N)
    log_likelihood_records = zeros(max_n_iteration+1)
    difference_records = zeros(Float64, max_n_iteration, N)

    p_prev = p0   # Initialize equilibrium probablity density
    log_likelihood_records[1] = get_loglikelihood_delta(Nh, Np, xratio, xavg, p_prev, D_guess_array, Nv, tau, y_record, save_freq)

    p_container[1, :] = p0 # The first row in container is p0
    println("Start EM...")
    for iter_id = 1:max_n_iteration
        println(@sprintf "EM Iteration-ID: %d" iter_id)
        p_em, log_likelihood = forward_backward_delta(Nh, Np, xratio, xavg, p_prev, D_guess_array, Nv, tau, y_record, save_freq)
        p_em = max.(p_em, 1e-10)
        
        # smooth
        window_size = 7
        order = 3
        sg_result = savitzky_golay(p_em, window_size, order)
        p_em_smooth = max.(sg_result.y, 1e-10)
        p_em_smooth = p_em_smooth / sum(w0 .* p_em_smooth)
        p_prev[:,1] = p_em_smooth

        # calculate difference_between smooth and original em result
        difference_records[iter_id,:] = (p_em_smooth - p_em) .^ 2

        # Record peq, D, log_likelihood
        p_container[iter_id+1, :] = p_em_smooth
        log_likelihood_records[iter_id+1] = get_loglikelihood_delta(Nh, Np, xratio, xavg, p_prev, D_guess_array, Nv, tau, y_record, save_freq)
    end
    println("End EM...")
    return p_container, log_likelihood_records, difference_records
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
        "--max_n_iteration"
            help = "max of iteration used in learning peq (m)"
            arg_type = Int
            default = 10
        "--D_guess"
            help = "The guess of diffusion coefficient"
            arg_type = Float64
            default = 40.
        "--sele_iter_id"
            help = "Selected m (the result of learning peq) (optional)"
            arg_type = Int
            default = 10
    end
    return parse_args(s)
end

function main()
    @show parsed_args = parse_commandline()
    host = parsed_args["host"]
    iter_idx_n = parsed_args["iter_idx_n"]
    n_frames = parsed_args["n_frames"]
    max_n_iteration = parsed_args["max_n_iteration"]
    D_guess = parsed_args["D_guess"]
    sele_iter_id = parsed_args["sele_iter_id"] # only active when n != 0

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
    data_folder = joinpath(work_dir, "data", host)
    hdf5_name = @sprintf "traj.%s.frames.hdf5" n_frames
    f_in = joinpath(data_folder, hdf5_name)
    temp = load(f_in, "distance")
    t_record = load(f_in, "time")
    y_record = zeros(tau, 1)
    y_record[:,1] = temp
    #--------------------------------------------------------------------------

    #--- Set initial guess of peq ---------------------------------------------
    if iter_idx_n == 0
        V0 = potential_uniform.(xref)
        rho_0 = get_rhoeq(V0, w0)
        p0 = rho_0 .* rho_0 # initial guess
    else
        jld_name = @sprintf "peq_update_record_n%s_%s.jld" iter_idx_n-1 n_frames
        f_in = joinpath(data_folder, jld_name)
        p_container = load(f_in, "p_container")
        p0 = zeros(N, 1)
        p0[:,1] = p_container[sele_iter_id+1, :]
    end
    #--------------------------------------------------------------------------

    #--- EM -------------------------------------------------------------------
    D_guess_array = D_guess * ones(Nv)
    p_container, log_likelihood_records, difference_records = expectation_maximization(max_n_iteration, N, Nh, Np, xratio, xavg, Nv, tau, y_record, save_freq, p0, D_guess_array, w0)
    #--------------------------------------------------------------------------

    #------ Output ------------------------------------------------------------
    jld_name = @sprintf "peq_update_record_n%s_%s.jld" iter_idx_n n_frames
    f_out_pcontain = joinpath(data_folder, jld_name)

    jld_name = @sprintf "log_likelihood_n%s_%s.jld" iter_idx_n n_frames
    f_out_l_record = joinpath(data_folder, jld_name)

    jld_name = @sprintf "smooth_difference_mat_n%s_%s.jld" iter_idx_n n_frames
    f_out_diff = joinpath(data_folder, jld_name)

    save(f_out_pcontain, "xref", xref, "p_container", p_container, "D_guess", D_guess)
    println(@sprintf "Write p_container to %s" f_out_pcontain)

    save(f_out_l_record, "log_likelihood_records", log_likelihood_records)
    println(@sprintf "Write log_likelihood_records to %s" f_out_l_record)

    save(f_out_diff, "difference_records", difference_records)
    println(@sprintf "Write difference_records to %s" f_out_diff)
    #--------------------------------------------------------------------------
end

main()