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

ErrorNorm(A, Aans, A0) = sqrt(sum((A - Aans) .^ 2) / sum((Aans - A0) .^ 2))

Setting_num = 6
simulation_name = "Vanila_parameter"

dir = "Comparison_SomeSetting/Noise_dynamics/Setting$Setting_num"


n_dim = true # システム同定の際に状態空間の次元を情報として与えるか

Trials = 20
Num_systems = 100

struct Problem_param
    Q1
    Q2
    Q_prime
    last_value
end


## ゲイン最適化アルゴリズムのパラメータ
eta = 0.005 # 0.05だといい結果が出そう
epsilon_GD = 1e-16
epsilon_EstGrad = 0
delta = 0 # 確率のパラメータ
eps_interval = 0.3
M_interval = 5

#norm_omega = sqrt(2 * system.p) * M_interval

N_sample = 5 # 50
N_GD = 100 # 200
N_inner_obj = 20 #20
tau = 2000
r = 0.1

method_num = 3
method_names_list = ["Onepoint_SimpleBaseline", "One_point_WithoutBase", "TwoPoint"]
method_name = method_names_list[method_num]
## 最適化のパラメータ設定
mutable struct Optimization_param
    eta
    epsilon_GD
    epsilon_EstGrad
    delta
    eps_interval
    M_interval
    N_sample
    N_inner_obj
    N_GD
    tau
    r
    method_name
end

Opt = Optimization_param(
    eta,
    epsilon_GD,
    epsilon_EstGrad,#理論保証から導かれたパラメータを使用する時に使う
    delta, #理論保証から導かれたパラメータを使用する時に使う
    eps_interval,
    M_interval,
    N_sample,
    N_inner_obj,
    N_GD,
    tau,
    r,
    method_name
)

# FF推定のためのパラメータ
Kp_uhat_power = 0.3
epsilon_u = 1e-5

## システム同定パラメータ

Num_Samples_per_traj = 2 * N_inner_obj * N_sample * N_GD #200000 #1つの軌道につきサンプル数個数
println("Num_Samples_per_traj: ", Num_Samples_per_traj)
PE_power = 1 #Setting1~4までは20でやっていた．5は1
noise_free = false


## パラメータの保存
params = Dict(
    "N_inner_obj" => N_inner_obj,
    "eta" => eta,
    "epsilon_GD" => epsilon_GD,
    "epsilon_EstGrad" => epsilon_EstGrad,
    "delta" => delta,
    "eps_interval" => eps_interval,
    "M_interval" => M_interval,
    "N_sample" => N_sample,
    "N_GD" => N_GD,
    "tau" => tau,
    "r" => r,
    "method_name" => method_name,
    "Kp_uhat_power" => Kp_uhat_power,
    "epsilon_u" => epsilon_u,
    "Ts" => "system.hを採用",
    "Num_Samples_per_traj" => Num_Samples_per_traj,
    "PE_power" => PE_power,
    "Trials" => Trials,
    "Num_systems" => Num_systems,
    "date" => Dates.now(),
)

if !isdir(dir)
    mkdir(dir)  # フォルダを作成
end
dir = dir * "/" * simulation_name
if !isdir(dir)
    mkdir(dir)  # フォルダを作成
end

open(dir * "/params.json", "w") do io
    JSON.print(io, params, 4)  # 可読性も高く整形出力
end

# 数値実験開始

MFree_u_error_list = []
MBase_u_error_list = []

MBase_ObjMean_list = []
MFree_ObjMean_list = []

rng_parent = MersenneTwister(1)
for iter_system in 1:Num_systems
    println("iter_system: ", iter_system)
    #ランダムにシステムを生成
    seed_gen_system = rand(rng_parent, Int64)
    seed_attr_system = rand(rng_parent, Int64)
    system = Generate_system(seed_gen_system, seed_attr_system, Setting_num)

    #最適化問題のパラメータ
    Q1 = 200.0I(system.p)
    Q2 = 20.0I(system.p)
    Q_prime = [system.C'*Q1*system.C zeros(system.n, system.p); zeros(system.p, system.n) Q2]
    Q_prime = Symmetric((Q_prime + Q_prime') / 2)
    last_value = true

    K_P = 1.0I(system.p)
    K_I = 1.0I(system.p)
    prob = Problem_param(Q1, Q2, Q_prime, last_value)

    # FF推定のためのパラメータ
    K_P_uhat = Kp_uhat_power * I(system.p)

    ## システム同定のためのパラメータ
    Ts = system.h #サンプル間隔
    Num_trajectory = 1
    Steps_per_sample = Ts / system.h
    Steps_per_sample = round(Steps_per_sample)
    Num_TotalSamples = Num_trajectory * Num_Samples_per_traj
    T_Sysid = Ts * Num_Samples_per_traj

    Kp_MBase_last_list = []
    Ki_MBase_last_list = []
    ustar_MBase_list = []

    Kp_MFree_last_list = []
    Ki_MFree_last_list = []
    uhat_MFree_list = []

    for trial in 1:Trials
        println("iter_system: $iter_system trial: $trial ")

        ## システム同定
        est_system = Est_system(system, Num_TotalSamples, Num_trajectory, Steps_per_sample, Ts, T_Sysid, PE_power)
        Q_prime_sysid = [est_system.C'*Q1*est_system.C zeros(est_system.n, est_system.p); zeros(est_system.p, est_system.n) Q2]
        prob_sysid = Problem_param(Q1, Q2, Q_prime_sysid, last_value)
        Kp_seq_SysId, Ki_seq_SysId, _ = ProjGrad_Gain_Conststep_ModelBased_Noise(K_P, K_I, est_system, prob_sysid, Opt)
        push!(Kp_MBase_last_list, Kp_seq_SysId[end])
        push!(Ki_MBase_last_list, Ki_seq_SysId[end])
        push!(ustar_MBase_list, est_system.u_star)


        ## FF計算
        tau_u = Compute_tauu(system, K_P_uhat, epsilon_u)
        x_0 = rand(system.rng, system.Dist_x0, system.n)
        reset = zeros(system.m)
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
        u_hat = (Ematrix * Ematrix') \ error_zero
        u_hat = -Ematrix' * u_hat

        push!(uhat_MFree_list, u_hat)

        println("推定したフィードフォワード: ", u_hat)
        println("uの平衡点, 正解のフィードフォワード: ", system.u_star)
        println("Difference: ", u_hat - system.u_star)
        println("error: ", sqrt(sum((u_hat - system.u_star) .^ 2)))

        ## ゲイン最適化
        Kp_seq_MFree_uhat, Ki_seq_MFree_uhat, _ = ProjectGradient_Gain_Conststep_Noise(K_P, K_I, u_hat, system, prob, Opt)
        push!(Kp_MFree_last_list, Kp_seq_MFree_uhat[end])
        push!(Ki_MFree_last_list, Ki_seq_MFree_uhat[end])
    end

    # 結果の格納

    MFree_u_error_itersystem = []
    MBase_u_error_itersystem = []
    for trial in 1:Trials
        push!(MFree_u_error_itersystem, ErrorNorm(uhat_MFree_list[trial], system.u_star, zeros(system.m)))
        push!(MBase_u_error_itersystem, ErrorNorm(ustar_MBase_list[trial], system.u_star, zeros(system.m)))
    end

    MFree_Obj_itersystem = []
    MBase_Obj_itersystem = []
    for trial in 1:Trials
        push!(MFree_Obj_itersystem, ObjectiveFunction_noise(system, prob, Kp_MFree_last_list[trial], Ki_MFree_last_list[trial]))
        push!(MBase_Obj_itersystem, ObjectiveFunction_noise(system, prob, Kp_MBase_last_list[trial], Ki_MBase_last_list[trial]))
    end

    ## 各trialでのFF推定相対誤差の平均を格納する
    push!(MFree_u_error_list, mean(MFree_u_error_itersystem))
    push!(MBase_u_error_list, mean(MBase_u_error_itersystem))

    ## 各trialでの目的関数の平均を格納する
    push!(MFree_ObjMean_list, mean(MFree_Obj_itersystem))
    push!(MBase_ObjMean_list, mean(MBase_Obj_itersystem))

    println("mean of FF error(MFree): ", mean(MFree_u_error_itersystem))
    println("mean of FF error(MBase): ", mean(MBase_u_error_itersystem))

    println("mean of Optimized Obj(MFree): ", mean(MFree_Obj_itersystem))
    println("mean of Optimized Obj(MBase): ", mean(MBase_Obj_itersystem))

end


@save dir * "/MFree_u_error_list.jld2" MFree_u_error_list
@save dir * "/MBase_u_error_list.jld2" MBase_u_error_list
@save dir * "/MFree_ObjMean_list.jld2" MFree_ObjMean_list
@save dir * "/MBase_ObjMean_list.jld2" MBase_ObjMean_list
