using ControlSystemIdentification, ControlSystemsBase
using JLD2
using Plots
using Random

include("function_noise.jl")
include("function_orbit.jl")
#rng = MersenneTwister(1)
Setting_num = 7


n_dim = true
noise_free = false

@load "System_setting/Noise_dynamics/Settings/Setting$Setting_num/Settings.jld2" Setting

for (key, value) in Setting
    @eval $(Symbol(key)) = $value
end

Ts = 20 * system.h#サンプル間隔
Num_trajectory = 1 #サンプル数軌道の数
Num_Samples_per_traj = 4000000 #1つの軌道につきサンプル数個数
PE_power = 0.1 #Setting1~4までは20でやっていた．5は1


Steps_per_sample = Ts / system.h
Steps_per_sample = round(Steps_per_sample)
println("Steps_per_sample: ", Steps_per_sample)
Num_TotalSamples = Num_trajectory * Num_Samples_per_traj

Us = zeros(system.m, Num_TotalSamples)
Ys = zeros(system.p, Num_TotalSamples)
T = Ts * Num_Samples_per_traj

for i in 1:Num_trajectory
    x_0 = rand(system.rng, system.Dist_x0, system.n)
    i = Int(i)
    u_s, x_s, y_s, Timeline = Orbit_Identification_noise(system, x_0, T, PE_power=PE_power)
    if noise_free
        u_s, x_s, y_s, Timeline = Orbit_Identification_noiseFree(system, x_0, T)
    end
    println(y_s[1, 1:50])

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
println("Ys", Ys[1, 1:10])

# Inf やNanがないかの確認
bad = Dict{Tuple{Int,Int},Float64}()

for j in axes(Ys, 2), i in axes(Ys, 1)
    val = Ys[i, j]
    if !isfinite(val)
        bad[(i, j)] = val
    end
end

println(bad)  # 例: {(3,5)=>NaN, (12,1)=>Inf, ...}
# N/S比の推定
function hankel_blocks(Y, i)
    p, N = size(Y)
    cols = N - i + 1
    H = Matrix{eltype(Y)}(undef, p * i, cols)
    for k in 1:i
        H[(p*(k-1)+1):(p*k), :] = @view Y[:, k:(k+cols-1)]
    end
    return H
end

function snr_from_output_svd(Y, i, nx)
    H = hankel_blocks(Y, i)
    s = svdvals(H)             # s sorted desc
    if isnothing(nx)
        # 簡易に“曲がり角”を検出（ギャップ最大）
        gaps = s[1:end-1] ./ s[2:end]
        nx = argmax(gaps)      # 超ざっくり
    end
    Esig = sum(s[1:nx] .^ 2)
    Enoise = sum(s[nx+1:end] .^ 2)
    SNRdB = 10 * log10(Esig / Enoise)
    return SNRdB, nx, s
end
sSNRdB, nx, s = snr_from_output_svd(Ys, 50, system.n)
println("Top20 singular val of HankelY: ", s[1:20])

if !isdir("System_setting/Noise_dynamics/Settings/Setting$Setting_num/N4sid_ndim=$(n_dim)_Ts=$(Ts)_NumSample=$(Num_Samples_per_traj)")
    mkdir("System_setting/Noise_dynamics/Settings/Setting$Setting_num/N4sid_ndim=$(n_dim)_Ts=$(Ts)_NumSample=$(Num_Samples_per_traj)")  # フォルダを作成
end

dir = "System_setting/Noise_dynamics/Settings/Setting$Setting_num/N4sid_ndim=$(n_dim)_Ts=$(Ts)_NumSample=$(Num_Samples_per_traj)"

sysc_true = ss(system.A, system.B, system.C, zeros(system.p, system.m))
sysd_true = c2d(sysc_true, Ts)
#println("True continuous model \n", sysc_true)
#println("True discrete model \n", sysd_true)

Data = iddata(Ys, Us, Ts)
Ys = nothing
Us = nothing
GC.gc()

# N4SID手法
#とりあえずシステムの次元は設定しない．
if n_dim
    sys = n4sid(Data, system.n, verbose=false, zeroD=true)
else
    sys = n4sid(Data, verbose=false, zeroD=true)
end
sys = n4sid(Data, system.n, verbose=true, zeroD=true)

method_name = "N4SID"
println("System Identification has done")
p = bodeplot([sysd_true[1, 1], sys.sys[1, 1]], label=["True" "n4sid"], layout=(2, 1))
savefig(p, dir * "/Disc_Bode_N4SID_11.png")
p = bodeplot([sysd_true[1, 2], sys.sys[1, 2]], label=["True" "n4sid"], layout=(2, 1))
savefig(p, dir * "/Disc_Bode_N4SID_12.png")
p = bodeplot([sysd_true[2, 1], sys.sys[2, 1]], label=["True" "n4sid"], layout=(2, 1))
savefig(p, dir * "/Disc_Bode_N4SID_21.png")
p = bodeplot([sysd_true[2, 2], sys.sys[2, 2]], label=["True" "n4sid"], layout=(2, 1))
savefig(p, dir * "/Disc_Bode_N4SID_22.png")


#println("estimated model \n", cont_sys)
#println("sys.sys: ", cont_sys.sys)
#println("sys.Q: ", cont_sys.Q)
#println("sys.R: ", cont_sys.R)
cont_sys = d2c(sys)
A_est, B_est, C_est, D_est = cont_sys.A, cont_sys.B, cont_sys.C, cont_sys.D
W_est, V_est = cont_sys.Q, cont_sys.R

if !isdir("test_file/Gain_result_noise/Settings/Setting$Setting_num")
    mkdir("test_file/Gain_result_noise/Settings/Setting$Setting_num")  # フォルダを作成
end
p = bodeplot([sysc_true[1, 1], cont_sys.sys[1, 1]], label=["True" "n4sid"], layout=(2, 1))
savefig(p, dir * "/Cont_Bode_Ns4SID_11.png")
p = bodeplot([sysc_true[1, 2], cont_sys.sys[1, 2]], label=["True" "n4sid"], layout=(2, 1))
savefig(p, dir * "/Cont_Bode_N4SID_12.png")
p = bodeplot([sysc_true[2, 1], cont_sys.sys[2, 1]], label=["True" "n4sid"], layout=(2, 1))
savefig(p, dir * "/Cont_Bode_N4SID_21.png")
p = bodeplot([sysc_true[2, 2], cont_sys.sys[2, 2]], label=["True" "n4sid"], layout=(2, 1))
savefig(p, dir * "/Cont_Bode_N4SID_22.png")


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

println(eigmin(W_est))
println(eigmin(V_est))

V_half_est = cholesky(V_est).L
W_half_est = cholesky(W_est).L
rng_for_est = MersenneTwister(1098)
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
    rng_for_est,
)



@save "System_setting/Noise_dynamics/Settings/Setting$Setting_num/N4sid_ndim=$(n_dim)_Ts=$(Ts)_NumSample=$(Num_Samples_per_traj)/Est_matrices.jld2" est_system

