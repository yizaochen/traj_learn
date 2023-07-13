import Pkg

work_dir = "/home/yizaochen/Projects/traj_learn/TrajLearn"
src_dir = joinpath(work_dir, "src")

Pkg.activate(work_dir)
include(joinpath(src_dir, "FEM.jl"))
include(joinpath(src_dir, "Potential.jl"))
using .FEM, .Potential
using Printf
using JLD

function run_langevin(tau, xavg, n_interval, kref, D, dt)
    println("Start simulation....")
    y_record = zeros(tau+1, 1)  # In HMM, observing variables Y(t)
    t_record = zeros(tau+1, 1)

    ypos = xavg # Initial Position, from mean position
    t = 0.

    # Simulation Start
    for tau_id = 1:tau
        if (tau_id % 500 == 0)
            println(@sprintf "Progress: %2.2f" (tau_id / tau))
        end

        y_record[tau_id] = ypos
        t_record[tau_id] = t
        for i = 1:n_interval
            F = force_harmonic_well_k_mean(ypos, kref, xavg)
            ypos = ypos + D*dt*F + (dt*2*D)^(1/2) * randn()
            t += dt   
        end
    end
    y_record[tau+1] = ypos
    t_record[tau+1] = t
    println("End simulation....")
    return y_record, t_record
end

#--- Define host -------------
host = "harmonic_well"
#-----------------------------

#--- Get collocation points and integration kernel for whole domain ---
Nh = 64    # The number of Spectral element
Np = 4     # The order of polynomial which used to interpolate and integration
Nv = 72    # Number of eigenvectors

# Define Physical Domain
xratio = 64. # unit: L
xavg = 0.    # unit: L

x, w, Ldx, L = getLagrange(Np, xratio/Nh)
e_norm = x[end] - x[1]
interpo_xs = x .+ x[end]
N, xref, w0, Ldx, w = get_fem_xref_weights_basis(Nh, Np, xratio, xavg)
#-----------------------------------------------------------------------

#--- Set V(x) and peq(x) -----------------------------------------------
# More detail in ./adjust_double_well.jl (a Pluto notebook)
save_peq_potential = true

sigma_ref = 8. # ad hoc, unit: L
kref = get_k_by_sigma(sigma_ref)
Vref = harmonic_well_k_mean(xref, kref, xavg)
rho_eq = get_rhoeq(Vref, w0)
p_eq = rho_eq .* rho_eq

data_folder = joinpath(work_dir, "data", host)
if save_peq_potential
    jld_out = joinpath(data_folder, "potential_peq.jld")    
    save(jld_out, "xref", xref, "p_eq", p_eq, "Vref", Vref)
    println(@sprintf "Save potential and peq to %s" jld_out)
end
#------------------------------------------------------------------------

#---- Simulation --------------------------------------------------------
run_simulation = true

if run_simulation
    #--- Set physical parameters for simulation -----------------------------
    save_freq = 1e-1  # unit: T, 1T = 0.1 ns
    total_times = 100000 # unit: T, total: 10 us # 100000
    dt = 2e-5 # unit: T  # Integration Times
    D = 20.; # Diffusion coefficient, unit: L^{2} T^{-1}
    tau = Int(round(total_times / save_freq, digits=0))  # Number of photons
    n_interval = Int(round(save_freq / dt, digits=0));

    time_info = @sprintf "Save trajectory per %d timesteps. There will be %d data points." n_interval tau
    println(time_info)
    #------------------------------------------------------------------------

    #------------------Langenvin dynamics simulations------------------------
    y_record, t_record = run_langevin(tau, xavg, n_interval, kref, D, dt)
    #------------------------------------------------------------------------

    #--- Output Trajectory --------------------------------------------------
    jld_out = joinpath(data_folder, "traj.jld")
    save(jld_out, "y_record", y_record, "t_record", t_record, "D", D, "save_freq", save_freq)
    println(@sprintf "Save trajectory to %s" jld_out)
    #-----------------------------------------------------------------------
end
#------------------------------------------------------------------------