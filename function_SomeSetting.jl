include("function_noise.jl")
include("function_orbit.jl")
using ControlSystemIdentification, ControlSystemsBase
using Random

function Generate_system(seed_gen_system, seed_attr_system, Setting_num)
    rng_system = MersenneTwister(seed_gen_system)
    if Setting_num == 6
        n = 30
        m = 4
        p = m
        y_star = 5 * ones(p)

        J = 1.0 * randn(rng_system, Float64, (n, n))
        J = (J - J') / 2
        Randmat = 2.0 * randn(rng_system, Float64, (n, n))
        R = Randmat * Randmat'
        Hamilton = 1.0 * I(n)
        A = (J - R) * Hamilton
        B = 3 * randn(rng_system, Float64, (n, m))
        C = B' * Hamilton
        W = 0.01 * I(n)
        V = 0.0005 * I(p)
        h = 0.0005
        Dist_x0 = Uniform(-3, 3)
    elseif Setting_num == 7
        # Setting_num 6から共分散行列をランダムに
        n = 30
        m = 4
        p = m
        y_star = 5 * ones(p)

        J = 1.0 * randn(rng_system, Float64, (n, n))
        J = (J - J') / 2
        Randmat = 2.0 * randn(rng_system, Float64, (n, n))
        R = Randmat * Randmat'
        Hamilton = 1.0 * I(n)
        A = (J - R) * Hamilton
        B = 3 * randn(rng_system, Float64, (n, m))
        C = B' * Hamilton
        Randmat_w = randn(rng_system, Float64, (n, n))
        Rand_w = Randmat_w * Randmat_w'
        W = 0.001 * Rand_w
        Randmat_p = randn(rng_system, Float64, (p, p))
        Rand_p = Randmat_p * Randmat_p'
        V = 0.0001 * Rand_p
        h = 0.0005
        Dist_x0 = Uniform(-3, 3)
    elseif Setting_num == 10
        # Setting_num ６から次元数を落とす
        n = 20
        m = 2
        p = m
        y_star = 5 * ones(p)

        J = 1.0 * randn(rng_system, Float64, (n, n))
        J = (J - J') / 2
        Randmat = 2.0 * randn(rng_system, Float64, (n, n))
        R = Randmat * Randmat'
        Hamilton = 1.0 * I(n)
        A = (J - R) * Hamilton
        B = 3 * randn(rng_system, Float64, (n, m))
        C = B' * Hamilton
        W = 0.01 * I(n)
        V = 0.0005 * I(p)
        h = 0.0005
        Dist_x0 = Uniform(-3, 3)
    end

    equib = inv([A B; C zeros(p, m)]) * [zeros(n); y_star]
    x_star = equib[1:n]
    u_star = equib[(n+1):(n+p)]

    F = [A zeros((n, p)); -C zeros((p, p))]
    G = [B; zeros((p, m))]
    H = [C zeros(p, p); zeros(p, n) -I(p)]

    # 初期点の分布の計算
    Sigma0 = zeros((n), (n))
    Nsample = 10000
    for i in 1:Nsample
        #global Sigma0
        x_0 = rand(rng_system, Dist_x0, n)
        Sigma0 += (x_0 - x_star) * (x_0 - x_star)'
    end
    Sigma0 = Sigma0 / Nsample

    mean_ex0 = zeros(n)
    Nsample = 10000
    for i in 1:Nsample
        mean_ex0 += rand(rng_system, Dist_x0, n) - x_star
    end
    mean_ex0 /= Nsample

    V_half = cholesky(V).L
    W_half = cholesky(W).L

    rng_attr_system = MersenneTwister(seed_attr_system)

    system = TargetSystem(A,
        B,
        C,
        F,
        G,
        H,
        W,
        V,
        W_half,
        V_half,
        Dist_x0,
        h,
        y_star,
        x_star,
        u_star,
        Sigma0,
        mean_ex0,
        n,
        p,
        m,
        rng_attr_system)

    return system
end

function Est_system(system, Num_TotalSamples, Num_trajectory, Steps_per_sample, Ts, T_Sysid, PE_power)
    x_0 = rand(system.rng, system.Dist_x0, system.n)
    Us = zeros(system.m, Num_TotalSamples)
    Ys = zeros(system.p, Num_TotalSamples)
    @views for i in 1:Num_trajectory
        x_0 = rand(system.rng, system.Dist_x0, system.n)
        i = Int(i)
        u_s, x_s, y_s, Timeline = Orbit_Identification_noise(system, x_0, T_Sysid, PE_power=PE_power)
        if noise_free
            u_s, x_s, y_s, Timeline = Orbit_Identification_noiseFree(system, x_0, T_Sysid)
        end
        println("SDE has done")

        for j in 1:Num_Samples_per_traj
            j = Int(j)
            Us[:, Int((i - 1) * Num_Samples_per_traj + j)] .= u_s[:, Int((j - 1) * Steps_per_sample + 1)]
            Ys[:, Int((i - 1) * Num_Samples_per_traj + j)] .= y_s[:, Int((j - 1) * Steps_per_sample + 1)]
        end
        if i % 10 == 0
            println(i, " Samples collected.")
        end
    end
    println("Data has collected.")
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
    return est_system
end

function Est_discrete_system(system,
    Num_TotalSamples,
    Num_trajectory,
    Steps_per_sample,
    Ts,
    T_Sysid,
    PE_power;
    accuracy="Float64",
    true_dimension=false,
    h=0)
    # System identification for discrete system
    x_0 = rand(system.rng, system.Dist_x0, system.n)
    if accuracy == "Float64"
        Us, Ys = Orbit_Identification_noise_succinct(system,
            x_0,
            T_Sysid,
            Ts=Ts,
            PE_power=PE_power)
    elseif accuracy == "Float32"
        Us, Ys = Orbit_Identification_noise_Float32(system,
            x_0,
            T_Sysid,
            Ts=Ts,
            PE_power=PE_power, h=h)
    end

    #println("Data has collected.")
    #println(Ys[1, 1:10])
    #println(Ys[:, end-10:end])
    Data = iddata(Ys, Us, Ts)
    Ys = nothing
    Us = nothing
    #GC.gc()
    # N4sidによるシステム同定
    sys_disc = n4sid(Data, verbose=false, zeroD=true)
    #sys_disc = subspaceid(Data, system.n, verbose=false, zeroD=true)
    #println("System Identification has done")
    #Data = nothing
    #GC.gc()
    A_est_disc, B_est_disc, C_est_disc = sys_disc.A, sys_disc.B, sys_disc.C
    W_est_disc, V_est_disc = sys_disc.Q, sys_disc.R
    n_est = size(A_est_disc, 1)
    #println("estimeated dimention: ", n_est)
    m_est = size(B_est_disc, 2)
    p_est = size(C_est_disc, 1)
    equib_est = [(A_est_disc-I(n_est)) B_est_disc; C_est_disc zeros(p_est, m_est)] \ [zeros(n_est); system.y_star]
    x_star_est = equib_est[1:n_est]
    u_star_est = equib_est[(n_est+1):(n_est+p_est)]

    F_est_disc = [A_est_disc zeros((n_est, p_est)); -C_est_disc*A_est_disc I(p_est)]
    G_est_disc = [B_est_disc; zeros((p_est, m_est))]
    H_est_disc = [C_est_disc zeros(p_est, p_est); zeros(p_est, n_est) -I(p_est)]

    #ノイズの共分散行列が半正定値行列ならば正定値に補正してあげる
    if Real(eigmin(W_est_disc)) <= 0.0
        W_est_disc += (-Real(eigmin(W_est_disc)) + 1e-10) * I(size(W_est_disc, 1))
    end

    if Real(eigmin(V_est_disc)) <= 0.0
        V_est_disc += (-Real(eigmin(V_est_disc)) + 1e-10) * I(size(V_est_disc, 1))
    end

    #println("minimum eigvals of W_est: ", eigmin(W_est_disc))
    #println("minimum eigvals of V_est: ", eigmin(V_est_disc))
    V_half_est_disc = cholesky(V_est_disc).L
    W_half_est_disc = cholesky(W_est_disc).L
    est_system = TargetSystem(
        A_est_disc,
        B_est_disc,
        C_est_disc,
        F_est_disc,
        G_est_disc,
        H_est_disc,
        W_est_disc,
        V_est_disc,
        W_half_est_disc,
        V_half_est_disc,
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
    return est_system
end

function ZOH_discrete_system(system, Ts)
    A_disc = exp(system.A * Ts)
    B_disc = (exp(system.A * Ts) - I(system.n)) * (system.A \ system.B)

    equib_disc = [(A_disc-I(system.n)) B_disc; system.C zeros(system.p, system.m)] \ [zeros(system.n); system.y_star]
    x_star_disc = equib_disc[1:system.n]
    u_star_disc = equib_disc[(system.n+1):(system.n+system.p)]

    F_disc = [A_disc zeros((system.n, system.p)); -system.C*A_disc I(system.p)]
    G_disc = [B_disc; zeros((system.p, system.m))]
    H_disc = [system.C zeros(system.p, system.p); zeros(system.p, system.n) -I(system.p)]

    W_0 = lyap(system.A, system.W)
    W_h = lyap(system.A, exp(system.A * Ts) * system.W * exp(system.A' * Ts))
    W_disc = W_0 - W_h
    W_disc = (W_disc + W_disc') / 2

    #ノイズの共分散行列が半正定値行列ならば正定値に補正してあげる
    if Real(eigmin(W_disc)) <= 0.0
        W_disc += (-Real(eigmin(W_disc)) + 1e-10) * I(size(W_disc, 1))
    end

    println("minimum eigvals of W_disc: ", eigmin(W_disc))

    W_half_disc = cholesky(W_disc).L
    system_disc = TargetSystem(
        A_disc,
        B_disc,
        system.C,
        F_disc,
        G_disc,
        H_disc,
        W_disc,
        system.V,
        W_half_disc,
        system.V_half,
        system.Dist_x0,
        system.h,
        system.y_star,
        x_star_disc,
        u_star_disc,
        system.Sigma0,
        system.mean_ex0,
        system.n,
        system.p,
        system.m,
        system.rng,
    )
    return system_disc
end
