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
simulation_name = "FF_Vanila_parameter5_zoh"
estimated_param = false


@load "System_setting/Noise_dynamics/Settings/Setting$Setting_num/Settings.jld2" Setting


system = Setting["system"]

n_dim = true # システム同定の際に状態空間の次元を情報として与えるか

Trials = 20

# FF推定のためのパラメータ
K_P_uhat = 0.001 * I(system.p)
A_K_uhat = system.A - system.B * K_P_uhat * system.C
println(eigvals(A_K_uhat))
# tau_uのサイズの決定
epsilon_u = 1e-3
Z = lyap(A_K_uhat', I(system.n))
eigvals_Z = eigvals(Z)
eig_max_Z = maximum(eigvals_Z)
eig_min_Z = minimum(eigvals_Z)
tau_u = 2 * eig_max_Z * log(sqrt(system.m * system.p) * norm(system.C, 2) * norm(inv(A_K_uhat'), 2) * norm(system.B, 2) * eig_max_Z / (eig_min_Z * epsilon_u))
println("Estimated tau_u: ", tau_u)


## システム同定パラメータ
Ts = 0.005 #サンプル間隔
Num_trajectory = 1 #サンプル数軌道の数
PE_power = 20 #Setting1~4までは20でやっていた．5は1
Num_Samples_per_traj = (system.m + 1) * tau_u / Ts
Num_Samples_per_traj = Int(trunc(Num_Samples_per_traj))
println("Num_Samples_per_traj: ", Num_Samples_per_traj)
noise_free = false

Steps_per_sample = Ts / system.h
Steps_per_sample = round(Steps_per_sample)
println("Steps_per_sample: ", Steps_per_sample)
Num_TotalSamples = Num_trajectory * Num_Samples_per_traj
T_Sysid = Ts * Num_Samples_per_traj

## ディレクトリ作成
dir = "System_setting/Noise_dynamics/Settings/Setting$Setting_num/VS_ModelBase"
if !isdir(dir)
    mkdir(dir)  # フォルダを作成
end
dir = dir * "/" * simulation_name
if !isdir(dir)
    mkdir(dir)  # フォルダを作成
end

## パラメータの保存
params = Dict(
    "Trials" => Trials,
    "K_P_uhat" => K_P_uhat,
    "epsilon_u" => epsilon_u,
    "tau_u" => tau_u,
    "Ts" => Ts,
    "Num_trajectory" => Num_trajectory, #サンプル数軌道の数
    "Num_Samples_per_traj" => Num_Samples_per_traj,
    "PE_power" => PE_power,
    "Trials" => Trials,
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
    est_system = Est_discrete_system(system, Num_TotalSamples, Num_trajectory, Steps_per_sample, Ts, T_Sysid, PE_power)
    y_inf_sysid = -system.C * (system.A \ (system.B * est_system.u_star))
    println("yの誤差 モデルベース", system.y_star - y_inf_sysid)
    push!(list_ustar_Sysid, est_system.u_star)

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
    u_hat = -Ematrix' * ((Ematrix * Ematrix') \ error_zero)

    println("推定したフィードフォワード: ", u_hat)
    println("uの平衡点, 正解のフィードフォワード: ", system.u_star)
    println("Difference: ", u_hat - system.u_star)
    println("error: ", sqrt(sum((u_hat - system.u_star) .^ 2)))
    push!(list_uhat, u_hat)
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
savefig(dir * "/FF_ustar_error_boxplot.png")

boxplot(list_error_ystar_MFree, label="Model Free")
boxplot!(list_error_ystar_Sysid, label="System Identification")
savefig(dir * "/FF_y_error_boxplot.png")

