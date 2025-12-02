using Random, Distributions
using ControlSystemIdentification, ControlSystemsBase
using LinearAlgebra
using Plots
using LaTeXStrings
using ControlSystems
using StatsPlots

include("function_noise.jl")
include("function_orbit.jl")
include("function_AlgoParam.jl")
include("function_SomeSetting.jl")
using JLD2
using JSON
using Dates

Setting_num = 6
simulation_name = "Vanila_parameter4_zoh"
estimated_param = false


@load "System_setting/Noise_dynamics/Settings/Setting$Setting_num/Settings.jld2" Setting
system = Setting["system"]

dir = "System_setting/Noise_dynamics/Settings/Setting$Setting_num/VS_ModelBase"

dir = dir * "/" * simulation_name

@load dir * "/list_est_system.jld2" list_est_system

est_system = list_est_system[1]
println(abs.(eigvals(est_system.A)))

params = JSON.parsefile(dir * "/params.json")

eta_discrete = 0.0001   # 0.000001
epsilon_GD_discrete = 1e-5
N_GD_discrete = 500000
projection_num = 3
projection_list = ["diag", "Eigvals", "Frobenius"]
projection = projection_list[projection_num]

K_P_disc = 0.01 * I(system.p) #zeros((system.m, system.p))
K_I_disc = 0.01 * I(system.p) #zeros((system.m, system.p))

Q1 = 0.01 * I(system.p)
Q2 = 0.001 * I(system.p)

eps_interval = 0
M_interval = 5

norm_omega = sqrt(2 * system.p) * M_interval

N_sample = NaN
delta = NaN
N_GD = NaN
N_inner_obj = NaN
tau = NaN
r = NaN
method_name = NaN
epsilon_EstGrad = NaN

params_Gain_sysid = Dict(
    "Q1" => Q1,
    "Q2" => Q2,
    "eta_discrete" => eta_discrete,
    "epsilon_GD_discrete" => epsilon_GD_discrete,
    "eps_interval" => eps_interval,
    "N_GD_discrete" => N_GD_discrete,
    "M_interval" => M_interval,
    "projection" => projection,
    "K_P_disc" => K_P_disc,
    "K_I_disc" => K_I_disc,
    "date" => Dates.now(),
)

open(dir * "/params_Gain_sysid.json", "w") do io
    JSON.print(io, params_Gain_sysid, 4)  # 可読性も高く整形出力
end

Opt_discrete = Optimization_param(
    eta_discrete,
    epsilon_GD_discrete,
    epsilon_EstGrad,
    delta, #理論保証から導かれたパラメータを使用する時に使う
    eps_interval,
    M_interval,
    N_sample,
    N_inner_obj,
    N_GD_discrete,
    tau,
    r,
    projection,
    method_name
)

Q_prime = [est_system.C'*Q1*est_system.C zeros(est_system.n, system.p); zeros(system.p, est_system.n) Q2]
Q_prime = Symmetric((Q_prime + Q_prime') / 2)
last_value = true
struct Problem_param
    Q1
    Q2
    Q_prime
    last_value
end
prob_sysid = Problem_param(Q1, Q2, Q_prime, last_value)
perturb = 1e-5
rand_perturb = randn(est_system.rng, (system.m, system.p))
grad_P, grad_I = grad_discrete_noise(est_system, prob_sysid, K_P_disc, K_I_disc)
println("true: ", sum(grad_P .* rand_perturb))
obj = ObjectiveFunction_discrete_noise(est_system, prob_sysid, K_P_disc, K_I_disc)
obj_perturb = ObjectiveFunction_discrete_noise(est_system, prob_sysid, K_P_disc + perturb * rand_perturb, K_I_disc)
println("numerical derivative: ", (obj_perturb - obj) / perturb)

Trials = 20
list_Kp_Sysid = []
list_Ki_Sysid = []
for trial in 1:Trials
    ## 最適化問題のパラメータ
    println("trial: ", trial)
    est_system = list_est_system[trial]
    println("開ループ固有値絶対値最大値", maximum(abs.(eigvals(est_system.A))))
    Q_prime = [est_system.C'*Q1*est_system.C zeros(est_system.n, system.p); zeros(system.p, est_system.n) Q2]
    Q_prime = Symmetric((Q_prime + Q_prime') / 2)
    last_value = true
    struct Problem_param
        Q1
        Q2
        Q_prime
        last_value
    end
    prob_sysid = Problem_param(Q1, Q2, Q_prime, last_value)
    Kp_seq_SysId, Ki_seq_SysId, _ = ProjGrad_Discrete_Conststep_ModelBased_Noise(K_P_disc,
        K_I_disc,
        est_system,
        prob_sysid,
        Opt_discrete)
    push!(list_Kp_Sysid, Kp_seq_SysId[end])
    push!(list_Ki_Sysid, Ki_seq_SysId[end])
end

@save dir * "/list_Kp_Sysid.jld2" list_Kp_Sysid
@save dir * "/list_Ki_Sysid.jld2" list_Ki_Sysid