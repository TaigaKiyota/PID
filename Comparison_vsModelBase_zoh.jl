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


@load "System_setting/Noise_dynamics/Settings/Setting$Setting_num/Settings.jld2" Setting


system = Setting["system"]

n_dim = true # システム同定の際に状態空間の次元を情報として与えるか

Trials = 20
## 初期点のゲイン
K_P = 1.0I(system.p)
K_I = 1.0I(system.p)
K_P_disc = 0.0 * I(system.p)
K_I_disc = 0.0 * I(system.p)

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

## アルゴリズムのパラメータ
eta = 0.002 # 0.05だといい結果が出そう
epsilon_GD = 1e-16
delta = 0.05 # 確率のパラメータ
eps_interval = 0
M_interval = 5

norm_omega = sqrt(2 * system.p) * M_interval

N_sample = 25 # 50
N_GD = 30 # 200
N_inner_obj = 25 #20
tau = 15
r = 0.09

# FF推定のためのパラメータ
K_P_uhat = 0.001 * I(system.p)
A_K_uhat = system.A - system.B * K_P_uhat * system.C
println(eigvals(A_K_uhat))
# tau_uのサイズの決定
epsilon_u = 1e-3
tau_u = Compute_tauu(system, K_P_uhat, epsilon_u)
println("Estimated tau_u: ", tau_u)

# 理論保証によって導かれたアルゴリズムのパラメータ
obj_init = ObjectiveFunction_noise(system, prob, K_P, K_I)
stab = stability_margin(system, K_P, K_I) #初期点の安定余裕できんじ
epsilon_EstGrad = 1e-1
if estimated_param
    r, tau, tau_u, N_sample, N_inner_obj = Algo_params(system,
        prob,
        epsilon_EstGrad,
        obj_init,
        delta,
        norm_omega,
        K_P_uhat,
        stab)
    println("理論保証から導かれたパラメータ")
else
    println("手動で決定したパラメータ")
end
println("r: ", r)
println("tau: ", tau)
println("tau_u: ", tau_u)
println("N_sample: ", N_sample)
println("N_inner_obj: ", N_inner_obj)

method_num = 3
method_names_list = ["Onepoint_SimpleBaseline", "One_point_WithoutBase", "TwoPoint"]
method_name = method_names_list[method_num]

projection_num = 3
projection_list = ["diag", "Eigvals", "Frobenius"]
projection = projection_list[projection_num]
## 最適化のパラメータ設定

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

eta_discrete = 0.0001
epsilon_GD_discrete = 1e-5
N_GD_discrete = 20000

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



## システム同定パラメータ
Ts = 0.005 #サンプル間隔
Num_trajectory = 1 #サンプル数軌道の数
PE_power = 20 #Setting1~4までは20でやっていた．5は1
accuracy = "Float32"
#Num_Samples_per_traj = Num_Samples_per_traj = 2 * N_inner_obj * N_sample * N_GD #200000 #1つの軌道につきサンプル数個数
Num_Samples_per_traj = (2 * N_inner_obj * N_sample * N_GD * tau + tau_u) / Ts
#Num_Samples_per_traj = 5000
Num_Samples_per_traj = Int(trunc(Num_Samples_per_traj))
println("Num_Samples_per_traj: ", Num_Samples_per_traj)
noise_free = false

Steps_per_sample = Ts / system.h
Steps_per_sample = round(Steps_per_sample)
println("Steps_per_sample: ", Steps_per_sample)
Num_TotalSamples = Num_trajectory * Num_Samples_per_traj
T_Sysid = Ts * Num_Samples_per_traj
println("Identification horizon: ", T_Sysid)

## ディレクトリ作成
dir = "System_setting/Noise_dynamics/Settings/Setting$Setting_num/VS_ModelBase"
if !isdir(dir)
    mkdir(dir)  # フォルダを作成
end
dir = dir * "/" * simulation_name
if !isdir(dir)
    mkdir(dir)  # フォルダを作成
end


# 結果の参照のためのパラメータ
tau_eval = 400
Iteration_obj_eval = 200

## パラメータの保存
params = Dict(
    "Q1" => Q1,
    "Q2" => Q2,
    "N_inner_obj" => N_inner_obj,
    "eta" => eta,
    "eta_discrete" => eta_discrete,
    "epsilon_GD" => epsilon_GD,
    "epsilon_GD_discrete" => epsilon_GD_discrete,
    "epsilon_EstGrad" => epsilon_EstGrad,
    "delta" => delta,
    "eps_interval" => eps_interval,
    "M_interval" => M_interval,
    "N_sample" => N_sample,
    "N_GD" => N_GD,
    "N_GD_discrete" => N_GD_discrete,
    "tau" => tau,
    "r" => r,
    "method_name" => method_name,
    "projection" => projection,
    "K_P_uhat" => K_P_uhat,
    "epsilon_u" => epsilon_u,
    "Ts" => Ts,
    "Num_trajectory" => Num_trajectory, #サンプル数軌道の数
    "Num_Samples_per_traj" => Num_Samples_per_traj,
    "PE_power" => PE_power,
    "accuracy" => accuracy,
    "tau_eval" => tau_eval,
    "iteration_obj_eval" => Iteration_obj_eval,
    "K_P" => K_P,
    "K_I" => K_I,
    "K_P_disc" => K_P_disc,
    "K_I_disc" => K_I_disc,
    "date" => Dates.now(),
)

open(dir * "/params.json", "w") do io
    JSON.print(io, params, 4)  # 可読性も高く整形出力
end

list_ustar_Sysid = []
list_uhat = []

list_y_target_error = []
list_y_target_error_SysId = []

list_Kp_seq_Sysid = []
list_Ki_seq_Sysid = []

list_Kp_seq_ModelFree = []
list_Ki_seq_ModelFree = []

for trial in 1:Trials
    println("trial: ", trial)
    ## システム同定
    est_system = Est_discrete_system(system,
        Num_TotalSamples,
        Num_trajectory,
        Steps_per_sample,
        Ts,
        T_Sysid,
        PE_power,
        accuracy=accuracy)
    # システム同定によるゲイン最適化
    Q_prime_sysid = [est_system.C'*Q1*est_system.C zeros(est_system.n, est_system.p); zeros(est_system.p, est_system.n) Q2]
    prob_sysid = Problem_param(Q1, Q2, Q_prime_sysid, last_value)
    Kp_seq_SysId, Ki_seq_SysId, _ = ProjGrad_Discrete_Conststep_ModelBased_Noise(K_P_disc,
        K_I_disc,
        est_system,
        prob_sysid,
        Opt_discrete)
    push!(list_Kp_seq_Sysid, Kp_seq_SysId)
    push!(list_Ki_seq_Sysid, Ki_seq_SysId)
    push!(list_ustar_Sysid, est_system.u_star)
    y_inf_sysid = -system.C * (system.A \ (system.B * est_system.u_star))
    println("yの誤差 モデルベース", system.y_star - y_inf_sysid)

    ## FF 推定
    #ベースとなる誤差の収集
    x_0 = rand(system.rng, system.Dist_x0, system.n)
    reset = zeros(system.m)
    y = Error_t_P_noise(system, K_P_uhat, tau_u, reset)
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

    println("推定したフィードフォワード: ", u_hat)
    println("uの平衡点, 正解のフィードフォワード: ", system.u_star)
    println("Difference: ", u_hat - system.u_star)
    println("error: ", sqrt(sum((u_hat - system.u_star) .^ 2)))
    push!(list_uhat, u_hat)

    ## モデルフリー最適化
    Kp_seq_MFree_uhat, Ki_seq_MFree_uhat, _ = ProjectGradient_Gain_Conststep_Noise(K_P, K_I, u_hat, system, prob, Opt)
    push!(list_Kp_seq_ModelFree, Kp_seq_MFree_uhat)
    push!(list_Ki_seq_ModelFree, Ki_seq_MFree_uhat)
end
## 結果の保存
@save dir * "/list_ustar_Sysid.jld2" list_ustar_Sysid
@save dir * "/list_uhat.jld2" list_uhat
@save dir * "/list_Kp_seq_Sysid.jld2" list_Kp_seq_Sysid
@save dir * "/list_Ki_seq_Sysid.jld2" list_Ki_seq_Sysid
@save dir * "/list_Kp_seq_ModelFree.jld2" list_Kp_seq_ModelFree
@save dir * "/list_Ki_seq_ModelFree.jld2" list_Ki_seq_ModelFree
## 相対誤差を計算するもの　
ErrorNorm(A, Aans, A0) = sqrt(sum((A - Aans) .^ 2) / sum((Aans - A0) .^ 2))

## FF推定の結果表示
list_error_u_Sysid = []
list_error_u_MFree = []
for trial in 1:Trials
    push!(list_error_u_Sysid, ErrorNorm(list_ustar_Sysid[trial], system.u_star, zeros(system.m)))
    push!(list_error_u_MFree, ErrorNorm(list_uhat[trial], system.u_star, zeros(system.m)))
end

list_error_ystar_Sysid = []
list_error_ystar_MFree = []
for trial in 1:Trials
    ustar_sysid = list_ustar_Sysid[trial]
    y_inf_sysid = -system.C * (system.A \ (system.B * ustar_sysid))
    push!(list_error_ystar_Sysid, ErrorNorm(y_inf_sysid, system.y_star, zeros(system.m)))

    uhat = list_uhat[trial]
    y_inf_uhat = -system.C * (system.A \ (system.B * uhat))
    push!(list_error_ystar_MFree, ErrorNorm(y_inf_uhat, system.y_star, zeros(system.m)))
end

println(list_error_u_Sysid)
println(list_error_u_MFree)

println("yの相対誤差 提案手法", list_error_ystar_MFree)
println("yの相対誤差 モデルベース", list_error_ystar_Sysid)

boxplot(list_error_u_MFree, label="Model Free")
boxplot!(list_error_u_Sysid, label="System Identification")
savefig(dir * "/FF_error_boxplot.png")


## ゲイン最適化の結果表示

list_obj_MFree = []
list_obj_SysId = []
for trial in 1:Trials
    Obj_MFree_uhat = obj_mean_continuous(system, prob, (list_Kp_seq_ModelFree[trial])[end], (list_Ki_seq_ModelFree[trial])[end],
        system.u_star, tau_eval, Iteration_obj_eval, h=5e-4)

    Obj_SysId = obj_mean_zoh(system, prob,
        (list_Kp_seq_Sysid[trial])[end], (list_Ki_seq_Sysid[trial])[end],
        system.u_star, Ts, tau_eval, Iteration_obj_eval, h=5e-4)
    push!(list_obj_MFree, Obj_MFree_uhat)
    push!(list_obj_SysId, Obj_SysId)
end

boxplot(list_obj_MFree, label="Model Free")
boxplot!(list_obj_SysId, label="System Identification")
savefig(dir * "/Gain_MeanObj_boxplot.png")

@save dir * "/list_obj_MFree.jld2" list_obj_MFree
@save dir * "/list_obj_SysId.jld2" list_obj_SysId
