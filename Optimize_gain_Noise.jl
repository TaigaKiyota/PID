using Random, Distributions
using LinearAlgebra
using Plots
using LaTeXStrings
using ControlSystems

include("function_noise.jl")
include("function_orbit.jl")
include("function_SomeSetting.jl")

using Zygote
using JLD2

Setting_num = 6

@load "System_setting/Noise_dynamics/Settings/Setting$Setting_num/Settings.jld2" Setting

dir_path = "System_setting/Noise_dynamics/Settings/Setting$Setting_num/test_Gain"
if !isdir(dir_path)
    mkdir(dir_path)  # フォルダを作成
end

system = Setting["system"]

# 初期点のゲイン
K_P = 2.0 * I(system.p)
K_I = 2.0 * I(system.p)

## 最適化問題のパラメータ
Q1 = 200.0I(system.p)
Q2 = 5.0I(system.p)
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

Ts = 10 * system.h
system_disc = ZOH_discrete_system(system, Ts)

#=
F_K = system_disc.F - system_disc.G * [K_P K_I] * system_disc.H
println("Eigvals of closed_loop_discrete: ", abs.(eigvals(F_K)))
obj_zoh = ObjectiveFunction_zoh_noise(system, system_disc, prob, K_P, K_I, Ts)
println("zoh objective: ", obj_zoh)
println("continuous objective: ", ObjectiveFunction_noise(system, prob, K_P, K_I))

tau_simulate = 200
Iteration_obj = 100
u_hat = system.u_star
#obj_zoh = obj_mean_zoh(system, prob, K_P, K_I, u_hat, Ts, tau_simulate, Iteration_obj)
println("シミュレーションベースの目的関数の計算")
#println("zoh objective: ", obj_zoh)
continuous_zoh = obj_mean_continuous(system, prob, K_P, K_I, u_hat, tau_simulate, Iteration_obj)
println("continuous objective: ", continuous_zoh)
=#

# アルゴリズムのパラメータ
eta = 0.005 # 0.05だといい結果が出そう
epsilon_GD = 0.0001
epsilon_EstGrad = 0.0001
eps_interval = 0.3
M_interval = 3
N_sample = 10
N_inner_obj = 20 #20
N_GD = 40
tau = 15
r = 0.09
delta = 0.01

method_num = 3
method_names_list = ["Onepoint_SimpleBaseline", "One_point_WithoutBase", "TwoPoint"]
method_name = method_names_list[method_num]

projection_num = 2
projection_list = ["diag", "Eigvals", "Frobenius"]
projection = projection_list[projection_num]

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

#Kp_list_MFree, Ki_list_MFree, _ = ProjectGradient_Gain_Conststep_Noise(K_P, K_I, system.u_star, system, prob, Opt)

# FF推定開始
K_P_uhat = 0.01 * I(system.p)
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

#ベースとなる誤差の収集
x_0 = rand(system.rng, system.Dist_x0, system.n)
reset = zeros(system.m)
y = Error_t_P_noise(system, K_P_uhat, tau_u, reset)
error_zero = Error_t_P_noise(system, K_P_uhat, tau_u, reset)

Ematrix = zeros(system.p, system.m)
Umatrix = Matrix(I, system.m, system.m)
#データの収集
for i in 1:system.m
    global Umatrix
    global Ematrix
    reset = Umatrix[:, i]
    Umatrix[:, i] = reset
    x_0 = rand(system.rng, system.Dist_x0, system.n)
    #_, y_s, Timeline, _ = Orbit_P_noise(system, K_P, x_0, tau_u, reset)
    error_i = Error_t_P_noise(system, K_P_uhat, tau_u, reset)

    Ematrix[:, i] = error_i - error_zero
end
#println(Ematrix)
println("Umatrix: ", Umatrix)
#フィードフォワードの推定
u_hat = -Ematrix' * inv(Ematrix * Ematrix') * error_zero

println("推定したフィードフォワード: ", u_hat)
println("uの平衡点, 正解のフィードフォワード: ", system.u_star)
println("Difference: ", u_hat - system.u_star)
println("error: ", sqrt(sum((u_hat - system.u_star) .^ 2)))

ErrorNorm(A, Aans, A0) = sqrt(sum((A - Aans) .^ 2) / sum((Aans - A0) .^ 2))
y_inf_uhat = -system.C * (system.A \ (system.B * u_hat))
println(ErrorNorm(y_inf_uhat, system.y_star, zeros(system.m)))


println("\n Optimization with u_hat")
Kp_list_MFree_uhat, Ki_list_MFree_uhat, _ = ProjectGradient_Gain_Conststep_Noise(K_P, K_I, u_hat, system, prob, Opt)
Kp_list_MBase_uhat, Ki_list_MBase_uhat, _ = ProjGrad_Gain_Conststep_ModelBased_Noise(K_P, K_I, system, prob, Opt)

K_P_Mfree = Kp_list_MFree_uhat[end]
K_I_Mfree = Ki_list_MFree_uhat[end]
println("ModelFree Kp: ", K_P_Mfree)
println("ModelFree Ki: ", K_I_Mfree)

K_P_MBase = Kp_list_MBase_uhat[end]
K_I_MBase = Ki_list_MBase_uhat[end]
println("ModelBase Kp: ", K_P_MBase)
println("ModelBase Ki: ", K_I_MBase)

h = 1e-5
T = 5
x_0 = zeros(system.n)
_, y_s_hat, _ = Orbit_continuous_PI(system, K_P_Mfree, K_I_Mfree, u_hat, x_0, T, h=h)
_, y_s_MBase, _ = Orbit_continuous_PI(system, K_P_MBase, K_I_MBase, system.u_star, x_0, T, h=h)
Timeline = 0:system.h:T

plotting = plot(legendfontsize=18, tickfontsize=15, guidefont=18)
plot!(plotting, Timeline[1:end], y_s_hat[1, 1:end], labels="Proposed Method", lw=1.8, lc=:red)
plot!(plotting, Timeline[1:end], y_s_MBase[1, 1:end], labels="ModelBase", lw=1.8, lc=:blue)
hline!(plotting, system.y_star, label=L"y^{\star}", lc=:black, lw=3)
xlims!(0, T)
#ylims!(-1, 6)
xlabel!("Time")
ylabel!(L"y")
savefig(plotting, dir_path * "/Compare_Gain_estimatedFF.png")

for i in 1:system.p
    plotting = plot(legendfontsize=15, tickfontsize=15, legend=:right, guidefont=22)
    plot!(plotting, Timeline[1:end], y_s_hat[i, 1:end], labels="Proposed Method", lw=2, lc=:red)
    plot!(plotting, Timeline[1:end], y_s_MBase[i, 1:end], labels="ModelBase", lw=2, lc=:blue)
    hline!(plotting, system.y_star, label=L"y^{\star}", lc=:black, lw=4)
    xlims!(0, T)
    #ylims!(-3, 8)
    xlabel!(L"t")
    ylabel!(L"y(t)")
    savefig(plotting, dir_path * "/Compare_Gain_component$(i).png")
end
#=
# システム同定+モデルベースアルゴリズム
## システム同定パラメータ
Ts = 10 * system.h #サンプル間隔
Num_trajectory = 1 #サンプル数軌道の数
PE_power = 20 #Setting1~4までは20でやっていた．5は1
#Num_Samples_per_traj = Num_Samples_per_traj = 2 * N_inner_obj * N_sample * N_GD #200000 #1つの軌道につきサンプル数個数
Num_Samples_per_traj = (2 * N_inner_obj * N_sample * N_GD * tau) / Ts
Num_Samples_per_traj = Int(trunc(Num_Samples_per_traj))
println("Num_Samples_per_traj: ", Num_Samples_per_traj)
noise_free = false

Steps_per_sample = Ts / system.h
Steps_per_sample = round(Steps_per_sample)
println("Steps_per_sample: ", Steps_per_sample)
Num_TotalSamples = Num_trajectory * Num_Samples_per_traj
T_Sysid = Ts * Num_Samples_per_traj
Ts = 0.01
Num_Samples_per_traj = 2 * N_inner_obj * N_sample * N_GD #200000
println("Num_Samples_per_traj: ", Num_Samples_per_traj)
Sysid_method = "N4sid"
est_system = Est_discrete_system(system, Num_TotalSamples, Num_trajectory, Steps_per_sample, Ts, T_Sysid, PE_power)
Q_prime_sysid = [est_system.C'*Q1*est_system.C zeros(est_system.n, est_system.p); zeros(est_system.p, est_system.n) Q2]
prob_sysid = Problem_param(Q1, Q2, Q_prime_sysid, last_value)

# 離散化システムを導入する
Kp_list_SysId, Ki_list_SysId, _ = ProjGrad_Discrete_Conststep_ModelBased_Noise(K_P, K_I, est_system, prob_sysid, Opt)



# 結果のプロット
dir = "System_setting/Noise_dynamics/Settings/Setting$Setting_num/Optimize_Gain"
if !isdir(dir)
    mkdir(dir)  # フォルダを作成
end
## 目的関数のプロット
Obj_list_MFree = []
Obj_list_MFree_uhat = []
Obj_list_SysId = []
for i in 1:Opt.N_GD
    #Obj_MFree = ObjectiveFunction_noise(system, prob, Kp_list_MFree[i], Ki_list_MFree[i])
    Obj_MFree_uhat = ObjectiveFunction_noise(system, prob, Kp_list_MFree_uhat[i], Ki_list_MFree_uhat[i])
    #Obj_SysId = ObjectiveFunction_noise(system, prob, Kp_list_SysId[i], Ki_list_SysId[i])
    #push!(Obj_list_MFree, Obj_MFree)
    push!(Obj_list_MFree_uhat, Obj_MFree_uhat)
    #push!(Obj_list_SysId, Obj_SysId)
end

plotting = plot(tickfontsize=15, guidefont=18)

#plot!(plotting, Obj_list_MFree, labels="Model-Free", lc=:red)
plot!(plotting, Obj_list_MFree_uhat, labels="Model-Free_uhat", lc=:green)
#plot!(plotting, Obj_list_SysId, labels="SysId + Model-Base", lc=:blue)
#ylims!(-1, 5)
xlabel!("Iteration")
ylabel!(L"f(K)")
savefig(plotting, dir * "/ObjVal_ndim=$(n_dim)_Ts=$(Ts)_IdSample=$(Num_Samples_per_traj)_Nsample=$(N_sample)_Ninner=$(N_inner_obj).png")
=#