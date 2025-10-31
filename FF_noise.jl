using Random, Distributions
using LinearAlgebra
using Plots
using LaTeXStrings
using ControlSystems

include("function_noise.jl")
include("function_orbit.jl")

using Zygote
using JLD2

Setting_num = 6

@load "System_setting/Noise_dynamics/Settings/Setting$Setting_num/Settings.jld2" Setting

dir_path = "System_setting/Noise_dynamics/Settings/Setting$Setting_num/test_FF"
if !isdir(dir_path)
    mkdir(dir_path)  # フォルダを作成
end

system = Setting["system"]


# FF推定開始
K_P = 0.3 * I(system.p)
A_K = system.A - system.B * K_P * system.C
println(eigvals(A_K))
# tau_uのサイズの決定
epsilon_u = 1e-5
Z = lyap(A_K', I(system.n))
eigvals_Z = eigvals(Z)
eig_max_Z = maximum(eigvals_Z)
eig_min_Z = minimum(eigvals_Z)
tau_u = 2 * eig_max_Z * log(sqrt(system.m * system.p) * norm(system.C, 2) * norm(inv(A_K'), 2) * norm(system.B, 2) * eig_max_Z / (eig_min_Z * epsilon_u))
println("Estimated tau_u: ", tau_u)

#ベースとなる誤差の収集
x_0 = rand(system.rng, system.Dist_x0, system.n)
reset = zeros(system.m)
y = Error_t_P_noise(system, K_P, tau_u, reset)
error_zero = Error_t_P_noise(system, K_P, tau_u, reset)

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
    error_i = Error_t_P_noise(system, K_P, tau_u, reset)

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

# システム同定をした場合との比較
Ts = 0.01
Num_Samples_per_traj = 200000
n_dim = true
@load "System_setting/Noise_dynamics/Settings/Setting$Setting_num/N4sid_ndim=$(n_dim)_Ts=$(Ts)_NumSample=$(Num_Samples_per_traj)/Est_matrices.jld2" est_system

println("システム同定したフィードフォワード: ", est_system.u_star)
println("uの平衡点, 正解のフィードフォワード: ", system.u_star)
println("Difference: ", est_system.u_star - system.u_star)
println("error: ", sqrt(sum((est_system.u_star - system.u_star) .^ 2)))

# 結果のプロット

if !isdir("Presentation/plot/Paper/Noise/DataMatrix")
    mkdir("Presentation/plot/Paper/Noise/DataMatrix")  # フォルダを作成
end

if !isdir("Presentation/plot/Paper/Noise/DataMatrix/Setting$(Setting_num)")
    mkdir("Presentation/plot/Paper/Noise/DataMatrix/Setting$(Setting_num)")  # フォルダを作成
end

reset_zero = zeros(system.p)
x_0 = zeros(system.n)
z_0 = zeros(system.p)
T = 100

# シミュレーションによる結果の確認
K_P = 1.0I(system.p)
K_I = 0.001 * I(system.p)

_, y_s_hat, _, Timeline, _, _ = Orbit_PI_noise(system, K_P, K_I, x_0, z_0, T, u_hat)
_, y_s_star, _, Timeline, _, _ = Orbit_PI_noise(system, K_P, K_I, x_0, z_0, T, system.u_star)
_, y_s_sysid, _, Timeline, _, _ = Orbit_PI_noise(system, K_P, K_I, x_0, z_0, T, est_system.u_star)

plotting = plot(legendfontsize=18, tickfontsize=15, guidefont=18)
plot!(plotting, Timeline[2:end], y_s_hat[1, 2:end], labels=L"u_0= \widehat{u}_0", lc=:red)
plot!(plotting, Timeline[2:end], y_s_sysid[1, 2:end], labels=L"u_0= \widehat{u}_{est}", lc=:green)
plot!(plotting, Timeline[2:end], y_s_star[1, 2:end], labels=L"u_0= u^{\star}", lc=:blue, ls=:dot)
hline!(plotting, system.y_star, label=L"y^{\star}", linestyle=:dot, lc=:black)
xlims!(0, T)
ylims!(-1, 6)
xlabel!("Time (s)")
ylabel!(L"y")
savefig(plotting, "Presentation/plot/Paper/Noise/DataMatrix/Setting$(Setting_num)/Compare_DataMatrix.png")

for i in 1:system.p
    plotting = plot(legendfontsize=18, tickfontsize=15, guidefont=22)
    plot!(plotting, Timeline[2:end], y_s_hat[i, 2:end], labels=L"u_0= \widehat{u}_0", lc=:red)
    plot!(plotting, Timeline[2:end], y_s_sysid[i, 2:end], labels=L"u_0= \widehat{u}_{est}", lc=:green)
    plot!(plotting, Timeline[2:end], y_s_star[i, 2:end], labels=L"u_0= u^{\star}", lc=:blue, ls=:dot)
    hline!(plotting, system.y_star, label=L"y^{\star}", linestyle=:dot, lc=:black)
    xlims!(0, T)
    ylims!(-3, 8)
    xlabel!(L"t")
    ylabel!(L"y(t)")
    savefig(plotting, "Presentation/plot/Paper/Noise/DataMatrix/Setting$(Setting_num)/Compare_DataMatrix_component$i.png")
end
