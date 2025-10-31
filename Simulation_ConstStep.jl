using Random
using LinearAlgebra
using Plots
using LaTeXStrings
using ControlSystems

include("function.jl")
include("function_orbit.jl")

rng = MersenneTwister(1)
using Zygote
using JLD2

n = 4
m = 2
p = m
J = randn(rng, Float64, (n, n))
J = (J - J') / 2
Randmat = randn(rng, Float64, (n, n))
Orth = qr(Randmat).Q
#R = Orth * diagm(rand(rng, 0:5, n)) * Orth'
#R = Symmetric(R)
R = Randmat * Randmat'
Hamilton = 2 * I(n)
A = (J - R) * Hamilton

B = randn(rng, Float64, (n, m))
C = B' * Hamilton

#ランクのチェック
println(rank([A B; C zeros(p, m)]))

F = [A zeros(n, p); -C zeros(p, p)]
G = [B; zeros(p, m)]
H = [C zeros(p, p); zeros(p, n) -I(p); C*A zeros(p, p)]

K_P = I(p)
K_I = I(p)
#K_I = zeros(p, p)
K_D = I(p)
F_K = computeF_K(F, G, H, K_P, K_I, K_D)

K_M = inv(I(p) + K_D * H[1:p, 1:n] * G[1:n, :])

#目標値の設定
y_ans = randn(rng, Float64, p)
equib = inv([A B; C zeros(p, m)]) * [zeros(n); y_ans]
x_equib = equib[1:n]
u_equib = equib[(n+1):(n+p)]

#リプシッツ，強凸性の定数の推定
A_K = A - B * K_P * C
D = -C * inv(A_K) * B
#リプシッツ
L = 2 * (svd(B).S[1]) .^ 2 * (eigvals(Hamilton)[end])^2 / (eigvals(A_K)[end])^2
println("リプシッツ定数： ", L)
mu = eigvals(D' * D)[1]
println("強凸 :", mu)

println("ステップサイズ上限：", 2 / (L + mu))

eta = 0.05

a_dec = sqrt(1 - (eta * mu * L) / (mu + L))
b_dec = sqrt(1 - (2 * eta * mu * L) / (mu + L))
println(((L + mu) / 2) * a_dec^2 / (4 * b_dec))
println((L + mu - eta * mu * L) / (8 * b_dec))

eps_reset = 1e-3

println("勾配推定 r上限", (L + mu - eta * mu * L / 2) * eps_reset / (8 * b_dec))


tau = 20
N_reset = 50
r_reset = 0.0001
reset = zeros(p)
N_GD = 3000
reset_list, f_list = estimate_reset_ConstStep(tau, A_K, B, C, reset, x_equib, u_equib, N_reset, r_reset, N_GD, eta)

println(reset_list[end])
println(u_equib)

u_reset = reset_list[end]

println("出力とu_starとの誤差", sum((u_reset - u_equib) .^ 2))

K_P = I(p)
K_I = 0.01 * I(p)
#K_I = zeros(p, p)
K_D = I(p)
reset_zero = zeros(m)

x_0 = 3 * ones(n)
T = 20
h = 0.001
#x_s, y_s, Timeline, ans_y = Orbit_PD(A, B, C, K_P, K_D, x_0, reset, y_ans, x_equib, u_equib, h, T)
#x_s, y_s_D0, Timeline, ans_y = Orbit_PD(A, B, C, K_P, zeros((p, p)), x_0, reset, y_ans, x_equib, u_equib, h, T)
x_s, y_s_uans, z_s, Timeline, ans_aug, ans_y_uans = Orbit(A, B, C, K_P, K_I, K_D, x_0, u_equib, x_equib, y_ans, h, T)
x_s, y_s_ur, z_s, Timeline, ans_aug, ans_y_ur = Orbit(A, B, C, K_P, K_I, K_D, x_0, u_reset, x_equib, y_ans, h, T)
x_s, y_s_r0, z_s, Timeline, ans_aug, ans_y_r0 = Orbit(A, B, C, K_P, K_I, K_D, x_0, reset_zero, x_equib, y_ans, h, T)
plotting = plot(legendfontsize=18, tickfontsize=15, guidefont=18)
plot!(plotting, Timeline[2:end], y_s_ur[1, 2:end], labels=L"u_0= \widehat{u}_0", lc=:red, lw=4.5)
plot!(plotting, Timeline[2:end], y_s_uans[1, 2:end], labels=L"u_0= u^{\star}", lc=:blue, ls=:dash, lw=4.5)
plot!(plotting, Timeline[2:end], y_s_r0[1, 2:end], labels=L"u_0= 0", lc=:green, lw=4.5)
hline!(plotting, [y_ans[1]], label=L"y^{\star}", linestyle=:dot, lc=:black, lw=4.5)
xlims!(0, T)
ylims!(-3, 5)
xlabel!("Time (s)")
ylabel!("output")
savefig(plotting, "Presentation/plot/Compare_reset_estimation_ConstStep.png")
img = load("Presentation/plot/Compare_reset_estimation_ConstStep.png")
save("Presentation/plot/Compare_reset_estimation_ConstStep.eps", img)

plotting = plot(legendfontsize=18, tickfontsize=15, guidefont=22, legend=false)
plot!(plotting, f_list, labels=L"error", linewidth=4.5)
#xlims!(0, 20)
#ylims!(1e-5, 1)
xlabel!("Iteration", fontsize=18)
ylabel!(L"g(u_0)", fontsize=22)
savefig(plotting, "Presentation/plot/Objective_resets_ConstStep.png")

img = load("Presentation/plot/Objective_resets_ConstStep.png")
save("Presentation/plot/Objective_resets_ConstStep.eps", img)

ErrorNorm(A, Aans, A0) = sqrt(sum((A - Aans) .^ 2) / sum((Aans - A0) .^ 2))

Itr = size(f_list, 1)
reset_errors = []
for i in 1:Itr
    error = ErrorNorm(reset_list[i], u_equib, zeros(m))
    push!(reset_errors, error)
end

plotting = plot(yscale=:log10, legendfontsize=18, tickfontsize=15, guidefont=18)
plot!(plotting, reset_errors, linewidth=2)
#xlims!(0, 20)
#ylims!(1e-5, 1)
xlabel!("Iteration", fontsize=18)
ylabel!(L"\|| e(\tau) \||^2", fontsize=18)
ylabel!(L"\log_{10}" * L"\frac{\| u_0 -u^\star  \|}{\|u^\star \|}", fontsize=18)
savefig(plotting, "Presentation/plot/errors_resets_ConstStep.png")



