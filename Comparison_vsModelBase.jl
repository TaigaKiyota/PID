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
using JLD2
using JSON
using Dates

Setting_num = 7
simulation_name = "Vanila_parameter"
estimated_param = false

@load "System_setting/Noise_dynamics/Settings/Setting$Setting_num/Settings.jld2" Setting


system = Setting["system"]

n_dim = true # システム同定の際に状態空間の次元を情報として与えるか

Trials = 20
## 初期点のゲイン
K_P = 1.0I(system.p)
K_I = 1.0I(system.p)

## 最適化問題のパラメータ
Q1 = 200.0I(system.p)
Q2 = 20.0I(system.p)
Q_prime = [system.C'*Q1*system.C zeros(system.n, system.p); zeros(system.p, system.n) Q2]
Q_prime = Symmetric((Q_prime + Q_prime') / 2)
last_value = true
N_inner_obj = 20 #20
struct Problem_param
    Q1
    Q2
    Q_prime
    last_value
end
prob = Problem_param(Q1, Q2, Q_prime, last_value)

## アルゴリズムのパラメータ
eta = 0.005 # 0.05だといい結果が出そう
epsilon_GD = 1e-16
delta = 0.05 # 確率のパラメータ
eps_interval = 0.3
M_interval = 5

norm_omega = sqrt(2 * system.p) * M_interval

N_sample = 5 # 50
N_GD = 100 # 200
tau = 2000
r = 0.1

# FF推定のためのパラメータ
K_P_uhat = 0.3 * I(system.p)
A_K_uhat = system.A - system.B * K_P_uhat * system.C
println(eigvals(A_K_uhat))
# tau_uのサイズの決定
epsilon_u = 1e-5
Z = lyap(A_K_uhat', I(system.n))
eigvals_Z = eigvals(Z)
eig_max_Z = maximum(eigvals_Z)
eig_min_Z = minimum(eigvals_Z)
tau_u = 2 * eig_max_Z * log(sqrt(system.m * system.p) * norm(system.C, 2) * norm(inv(A_K_uhat'), 2) * norm(system.B, 2) * eig_max_Z / (eig_min_Z * epsilon_u))
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
    epsilon_EstGrad,
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



## システム同定パラメータ
Ts = system.h #サンプル間隔
Num_trajectory = 1 #サンプル数軌道の数
Num_Samples_per_traj = Num_Samples_per_traj = 2 * N_inner_obj * N_sample * N_GD #200000 #1つの軌道につきサンプル数個数
println("Num_Samples_per_traj: ", Num_Samples_per_traj)
PE_power = 1 #Setting1~4までは20でやっていた．5は1
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
    "Q1" => Q1,
    "Q2" => Q2,
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
    "K_P_uhat" => K_P_uhat,
    "epsilon_u" => epsilon_u,
    "Ts" => Ts,
    "Num_trajectory" => Num_trajectory, #サンプル数軌道の数
    "Num_Samples_per_traj" => Num_Samples_per_traj,
    "PE_power" => PE_power,
    "date" => Dates.now(),
)

open(dir * "/params.json", "w") do io
    JSON.print(io, params, 4)  # 可読性も高く整形出力
end

list_ustar_Sysid = []
list_uhat = []
list_Kp_seq_Sysid = []
list_Ki_seq_Sysid = []

list_Kp_seq_ModelFree = []
list_Ki_seq_ModelFree = []

for trial in 1:Trials
    println("trial: ", trial)
    ## システム同定
    # データを集める
    Us = zeros(system.m, Num_TotalSamples)
    Ys = zeros(system.p, Num_TotalSamples)
    for i in 1:Num_trajectory
        x_0 = rand(system.rng, system.Dist_x0, system.n)
        i = Int(i)
        u_s, x_s, y_s, Timeline = Orbit_Identification_noise(system, x_0, T_Sysid, PE_power=PE_power)
        if noise_free
            u_s, x_s, y_s, Timeline = Orbit_Identification_noiseFree(system, x_0, T_Sysid)
        end

        for j in 1:Num_Samples_per_traj
            j = Int(j)
            Us[:, Int((i - 1) * Num_Samples_per_traj + j)] = u_s[:, Int((j - 1) * Steps_per_sample + 1)]
            Ys[:, Int((i - 1) * Num_Samples_per_traj + j)] = y_s[:, Int((j - 1) * Steps_per_sample + 1)]
        end
        if i % 10 == 0
            println(i, " Samples collected.")
        end
    end
    println("Data has collected.")
    println("Ys", Ys[:, 1:10])
    #バグがないか確認
    bad = Dict{Tuple{Int,Int},Float64}()
    for j in axes(Ys, 2), i in axes(Ys, 1)
        val = Ys[i, j]
        if !isfinite(val)
            bad[(i, j)] = val
        end
    end
    println(bad)
    bad = nothing
    Data = iddata(Ys, Us, Ts)
    Ys = nothing
    Us = nothing
    GC.gc()
    # N4sidによるシステム同定
    #if n_dim
    #   sys = subspaceid(Data, system.n, verbose=false, zeroD=true)
    #else
    #   sys = subspaceid(Data, verbose=false, zeroD=true)
    #end
    sys = n4sid(Data, system.n, verbose=false, zeroD=true)
    #sys = subspaceid(Data, system.n, verbose=false, zeroD=true)
    println("System Identification has done")
    Data = nothing
    GC.gc()
    cont_sys = d2c(sys)
    A_est, B_est, C_est, D_est = cont_sys.A, cont_sys.B, cont_sys.C, cont_sys.D
    W_est, V_est = cont_sys.Q, cont_sys.R
    n_est = size(A_est, 1)
    m_est = size(B_est, 2)
    p_est = size(C_est, 1)
    equib_est = [A_est B_est; C_est zeros(p_est, m_est)] \ [zeros(n_est); system.y_star]
    x_star_est = equib_est[1:n_est]
    u_star_est = equib_est[(n_est+1):(n_est+p_est)]

    F_est = [A_est zeros((n_est, p_est)); -C_est zeros((p_est, p_est))]
    G_est = [B_est; zeros((p_est, m_est))]
    H_est = [C_est zeros(p_est, p_est); zeros(p_est, n_est) -I(p_est)]

    #ノイズの共分散行列が半正定値行列ならば正定値に補正してあげる
    if Real(eigmin(W_est)) <= 0.0
        W_est += (-Real(eigmin(W_est)) + 1e-10) * I(size(W_est, 1))
    end

    if Real(eigmin(V_est)) <= 0.0
        V_est += (-Real(eigmin(V_est)) + 1e-10) * I(size(V_est, 1))
    end

    println("minimum eigvals of W_est: ", eigmin(W_est))
    println("minimum eigvals of V_est: ", eigmin(V_est))
    V_half_est = cholesky(V_est).L
    W_half_est = cholesky(W_est).L
    #seed_for_est = rand(system.rng)
    #seed_for_est = round(seed_for_est)
    #rng_for_est = MersenneTwister(seed_for_est) ##あまりよくない？
    est_system = TargetSystem(
        A_est,
        B_est,
        C_est,
        F_est,
        G_est,
        H_est,
        W_est,
        V_est,
        W_half_est,
        V_half_est,
        system.Dist_x0,
        system.h,
        system.y_star,
        x_star_est,
        u_star_est,
        system.Sigma0,
        system.mean_ex0,
        n_est,
        p_est,
        m_est,
        system.rng,
    )
    push!(list_ustar_Sysid, u_star_est)
    # システム同定によるゲイン最適化
    Q_prime_sysid = [est_system.C'*Q1*est_system.C zeros(est_system.n, est_system.p); zeros(est_system.p, est_system.n) Q2]
    prob_sysid = Problem_param(Q1, Q2, Q_prime_sysid, last_value)
    Kp_seq_SysId, Ki_seq_SysId, _ = ProjGrad_Gain_Conststep_ModelBased_Noise(K_P, K_I, est_system, prob_sysid, Opt)
    push!(list_Kp_seq_Sysid, Kp_seq_SysId)
    push!(list_Ki_seq_Sysid, Ki_seq_SysId)
    println("size of Gain seq: ", size(Kp_seq_SysId))

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

println(list_error_u_MFree)
println(list_error_u_Sysid)
boxplot(list_error_u_MFree, label="Model Free")
boxplot!(list_error_u_Sysid, label="System Identification")
savefig(dir * "/FF_error_boxplot.png")

## ゲイン最適化の結果表示

list_seq_obj_MFree_uhat = []
list_seq_obj_SysId = []
for trial in 1:Trials
    Obj_seq_MFree_uhat = []
    Obj_seq_SysId = []
    for i in 1:Opt.N_GD
        if i <= (size(list_Kp_seq_ModelFree[trial], 1))
            Obj_MFree_uhat = ObjectiveFunction_noise(system, prob, (list_Kp_seq_ModelFree[trial])[i], (list_Ki_seq_ModelFree[trial])[i])
        else
            Obj_MFree_uhat = ObjectiveFunction_noise(system, prob, (list_Kp_seq_ModelFree[trial])[end], (list_Ki_seq_ModelFree[trial])[end])
        end

        if i <= (size(list_Kp_seq_Sysid[trial], 1))
            Obj_SysId = ObjectiveFunction_noise(system, prob, (list_Kp_seq_Sysid[trial])[i], (list_Ki_seq_Sysid[trial])[i])
        else
            Obj_SysId = ObjectiveFunction_noise(system, prob, (list_Kp_seq_Sysid[trial])[end], (list_Ki_seq_Sysid[trial])[end])
        end
        push!(Obj_seq_MFree_uhat, Obj_MFree_uhat)
        push!(Obj_seq_SysId, Obj_SysId)
    end
    push!(list_seq_obj_MFree_uhat, Obj_seq_MFree_uhat)
    push!(list_seq_obj_SysId, Obj_seq_SysId)
end

Concat_Mfree_hat = reduce(hcat, [c for c in list_seq_obj_MFree_uhat])
Mean_Mfree_hat = mean(Concat_Mfree_hat; dims=2)[:]
std_Mfree_hat = std(Concat_Mfree_hat; dims=2)[:]

Concat_MBase_Sysid = reduce(hcat, [c for c in list_seq_obj_SysId])
Mean_MBase_Sysid = mean(Concat_MBase_Sysid; dims=2)[:]
std_MBase_Sysid = std(Concat_MBase_Sysid; dims=2)[:]

plotting = plot(legendfontsize=18, tickfontsize=15, guidefont=18)
# プロット（平均 ± 1σのリボン）
plot!(plotting, 1:Opt.N_GD, Mean_Mfree_hat, ribbon=std_Mfree_hat, label="Model Free")
plot!(plotting, 1:Opt.N_GD, Mean_MBase_Sysid, ribbon=std_MBase_Sysid, label="Model Base +SysId")
savefig(plotting, dir * "/Gain_ConvergenceCurve.png")

