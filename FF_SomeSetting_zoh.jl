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
## 相対誤差を計算するもの　
ErrorNorm(A, Aans, A0) = sqrt(sum((A - Aans) .^ 2) / sum((Aans - A0) .^ 2))

Setting_num = 10
simulation_name = "FF_Vanila_parameter_zoh"
estimated_param = false


simulation_name_param = "Vanila_parameter_zoh"
dir_comparison = "Comparison_SomeSetting/Noise_dynamics/Setting$Setting_num"
dir_comparison = dir_comparison * "/" * simulation_name_param
@load dir_comparison * "/Dict_original_systems.jld2" Dict_original_systems

dir_experiment_setting = "System_setting/Noise_dynamics/Settings/Setting$Setting_num/VS_ModelBase"
dir_experiment_setting = dir_experiment_setting * "/" * simulation_name
params = JSON.parsefile(dir_experiment_setting * "/params.json")
Ts = params["Ts"] #変えてもいいPIゲインと合わせる
Ts = 0.01

Trials = 10
num_of_systems = 10



## ディレクトリ作成
dir_result = "Comparison_SomeSetting/Noise_dynamics/Setting$Setting_num"

if !isdir(dir_result)
    mkdir(dir_result)  # フォルダを作成
end
dir_result = dir_result * "/" * simulation_name
if !isdir(dir_result)
    mkdir(dir_result)  # フォルダを作成
end


per_system_estimated_tauu = Vector{Float64}(undef, num_of_systems)
per_system_list_uhat = Vector{Vector{Vector{Float64}}}(undef, num_of_systems)
per_system_list_ustar_Sysid = Vector{Vector{Vector{Float64}}}(undef, num_of_systems)
for iter_system in 1:num_of_systems
    println("iter_system: ", iter_system)
    system = Dict_original_systems["system$iter_system"]
    K_P_uhat = 0.001 * I(system.p)
    A_K_uhat = system.A - system.B * K_P_uhat * system.C
    # tau_uのサイズの決定
    epsilon_u = 1e-3
    tau_u = Compute_tauu(system, K_P_uhat, epsilon_u)
    println("Estimated tau_u: ", tau_u)
    ## システム同定パラメータ

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

    per_system_estimated_tauu[iter_system] = tau_u

    list_uhat = Vector{Vector{Float64}}(undef, Trials)
    list_ustar_Sysid = Vector{Vector{Float64}}(undef, Trials)
    for trial in 1:Trials
        println("trial: ", trial)
        ## システム同定
        est_system = Est_discrete_system(system, Num_TotalSamples, Num_trajectory, Steps_per_sample, Ts, T_Sysid, PE_power)
        y_inf_sysid = -system.C * (system.A \ (system.B * est_system.u_star))
        list_ustar_Sysid[trial] = est_system.u_star

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
        list_uhat[trial] = u_hat
    end
    per_system_list_uhat[iter_system] = list_uhat
    per_system_list_ustar_Sysid[iter_system] = list_ustar_Sysid
end

## 結果の保存
@save dir_result * "/per_system_list_uhat.jld2" per_system_list_uhat
@save dir_result * "/per_system_list_ustar_Sysid.jld2" per_system_list_ustar_Sysid
@save dir_result * "/per_system_estimated_tauu.jld2" per_system_estimated_tauu


for iter_system in 1:num_of_systems
    system = Dict_original_systems["system$iter_system"]
    list_uhat = per_system_list_uhat[iter_system]
    list_ustar_Sysid = per_system_list_ustar_Sysid[iter_system]

    K_P = zeros(system.p, system.p)

    list_error_ystar_Sysid = Vector{Float64}(undef, Trials)
    list_error_ystar_MFree = Vector{Float64}(undef, Trials)
    for trial in 1:Trials
        ustar_sysid = list_ustar_Sysid[trial]
        y_inf_sysid = -system.C * (system.A \ (system.B * ustar_sysid))
        list_error_ystar_Sysid[trial] = ErrorNorm(y_inf_sysid, system.y_star, zeros(system.m))

        uhat = list_uhat[trial]
        y_inf_uhat = -system.C * (system.A \ (system.B * uhat))
        list_error_ystar_MFree[trial] = ErrorNorm(y_inf_uhat, system.y_star, zeros(system.m))
    end
    if !isdir(dir_result * "/system$iter_system")
        mkdir(dir_result * "/system$iter_system")  # フォルダを作成
    end
    boxplot(list_error_ystar_MFree,
        tickfontsize=15, yguidefont=font(15), legend=false, fillalpha=0.0, outliercolor=:white, markercolor=:white)
    boxplot!(list_error_ystar_Sysid, fillalpha=0.0, outliercolor=:white, markercolor=:white)
    xticks!((1:2, ["Proposed method", "Indirect approach"]))
    ylims!(0, 0.35)
    savefig(dir_result * "/system$iter_system/FF_y_error_boxplot.png")

    _, ys_MFree = Orbit_continuous_P(system, K_P, u_hat, x_0, T)
end


