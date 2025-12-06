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
simulation_name = "Vanila_parameter5_zoh"
estimated_param = false


println("simulation_name: ", simulation_name)

dir = "Comparison_SomeSetting/Noise_dynamics/Setting$Setting_num"
dir = dir * "/" * simulation_name
@load dir * "/Dict_original_systems.jld2" Dict_original_systems

dir_experiment_setting = "System_setting/Noise_dynamics/Settings/Setting$Setting_num/VS_ModelBase"
dir_experiment_setting = dir_experiment_setting * "/" * simulation_name
params = JSON.parsefile(dir_experiment_setting * "/params.json")


r = params["r"]
r = 0.09
eta = params["eta"]
eta = 0.001
N_inner_obj = params["N_inner_obj"] #変えない
N_sample = params["N_sample"] #変えない
tau = params["tau"] #変えない
N_GD = params["N_GD"] #変えない
epsilon_GD = params["epsilon_GD"]

eps_interval = 0
M_interval = 5

# 理論保証によって導かれたアルゴリズムのパラメータ

println("r: ", r)
println("tau: ", tau)
println("N_sample: ", N_sample)
println("N_inner_obj: ", N_inner_obj)

method_name = params["method_name"]
projection = params["projection"]
## 最適化のパラメータ設定

delta = NaN
epsilon_EstGrad = NaN
Opt = Optimization_param(
    eta,
    epsilon_GD,
    epsilon_EstGrad,
    delta, #理論保証から導かれたパラメータを使用する時に使う
    eps_interval,
    M_interval,
    N_sample,
    N_inner_obj,
    N_GD,
    tau,
    r,
    projection,
    method_name
)

Trials = 20
num_of_systems = 20

Dict_list_Kp_seq_ModelFree = Dict{Any,Any}()
Dict_list_Ki_seq_ModelFree = Dict{Any,Any}()
Dict_list_uhat = Dict{Any,Any}()
for iter_system in 1:num_of_systems
    println("iter_system: ", iter_system)
    #システム同定
    system = Dict_original_systems["system$iter_system"]

    ## 最適化問題のパラメータ
    Q1 = 200.0I(system.p)
    Q2 = 20.0I(system.p)
    Q_prime = [system.C'*Q1*system.C zeros(system.n, system.p); zeros(system.p, system.n) Q2]
    Q_prime = Symmetric((Q_prime + Q_prime') / 2)
    last_value = true
    struct Problem_param
        Q1
        Q2
        Q_prime
        last_value
    end
    prob = Problem_param(Q1, Q2, Q_prime, last_value)

    K_P = 1.0 * I(system.p)
    K_I = 1.0 * I(system.p)

    # FF推定のためのパラメータ
    K_P_uhat = 0.001 * I(system.p) #手動で決める？
    A_K_uhat = system.A - system.B * K_P_uhat * system.C
    println(eigvals(A_K_uhat))
    # tau_uのサイズの決定
    epsilon_u = 1e-3
    tau_u = Compute_tauu(system, K_P_uhat, epsilon_u)
    println("Estimated tau_u: ", tau_u)

    list_Kp_seq_ModelFree = []
    list_Ki_seq_ModelFree = []
    list_uhat = []
    for trial in 1:Trials
        ## 最適化問題のパラメータ
        println("trial: ", trial)
        ## FF 推定
        #ベースとなる誤差の収集
        x_0 = rand(system.rng, system.Dist_x0, system.n)
        reset = zeros(system.m)
        #y = Error_t_P_noise(system, K_P_uhat, tau_u, reset)
        error_zero = Error_t_P_noise(system, K_P_uhat, tau_u, reset)

        Ematrix = zeros(system.p, system.m)
        Umatrix = Matrix(I, system.m, system.m)
        #データの収集
        for i in 1:system.m
            #global Umatrix
            #global Ematrix
            reset = Umatrix[:, i]
            Umatrix[:, i] = reset
            x_0 = rand(system.rng, system.Dist_x0, system.n)
            #_, y_s, Timeline, _ = Orbit_P_noise(system, K_P, x_0, tau_u, reset)
            error_i = Error_t_P_noise(system, K_P_uhat, tau_u, reset)

            Ematrix[:, i] = error_i - error_zero
        end
        u_hat = -Ematrix' * inv(Ematrix * Ematrix') * error_zero
        push!(list_uhat, u_hat)

        ## モデルフリー最適化
        Kp_seq_MFree_uhat, Ki_seq_MFree_uhat, _ = ProjectGradient_Gain_Conststep_Noise(K_P, K_I, u_hat, system, prob, Opt)
        push!(list_Kp_seq_ModelFree, Kp_seq_MFree_uhat)
        push!(list_Ki_seq_ModelFree, Ki_seq_MFree_uhat)
    end
    Dict_list_Kp_seq_ModelFree["system$iter_system"] = list_Kp_seq_ModelFree
    Dict_list_Ki_seq_ModelFree["system$iter_system"] = list_Ki_seq_ModelFree
    Dict_list_uhat["system$iter_system"] = list_uhat
end

@save dir * "/Dict_list_uhat.jld2" Dict_list_uhat
@save dir * "/Dict_list_Kp_seq_ModelFree.jld2" Dict_list_Kp_seq_ModelFree
@save dir * "/Dict_list_Ki_seq_ModelFree.jld2" Dict_list_Ki_seq_ModelFree