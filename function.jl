function computeF_K(F, G, H, K_P, K_I, K_D)
    p = size(K_D, 1)
    n = size(F, 1) - p

    K_M = inv(I(p) + K_D * H[1:p, 1:n] * G[1:n, :])
    K_Pbar = K_M * K_P
    K_Ibar = K_M * K_I
    K_Dbar = K_M * K_D
    F_K = F - G * [K_Pbar K_Ibar K_Dbar] * H
    return F_K
end

function ObjectiveFunction(F, G, H, C_aug, Sigma, K_P, K_I, K_D)
    F_K = computeF_K(F, G, H, K_P, K_I, K_D)
    X = lyap(F_K, Sigma)
    return sum(X .* C_aug)
end

function ObjectiveFunction_FK(F_K, C_aug, Sigma)
    X = lyap(F_K, Sigma)
    return sum(X .* C_aug)
end


function grad(F, G, H, C_aug, Sigma, K_P, K_I, K_D)
    p = size(K_D, 1)
    n = size(F, 1) - p
    F_K = computeF_K(F, G, H, K_P, K_I, K_D)
    X = lyap(F_K, Sigma)
    Y = lyap(F_K', C_aug)
    K_M = inv(I(p) + K_D * H[:, 1:n] * G[1:n, :])
    Z = -G' * Y * X * H'
    KmZ = K_M' * Z
    grad_Kp = KmZ[:, 1:p]
    grad_Ki = KmZ[:, p+1:2*p]
    grad_Kd = KmZ[:, 2*p+1:3*p] - KmZ * [K_M * K_P K_M * K_I K_M * K_D]' * K_M' * (H[:, 1:n] * G[1:n, :])'
    return grad_Kp, grad_Ki, grad_Kd
end

function error_tau(tau, A_K, B, C, reset, x_0, x_equib, u_equib)
    err_state = inv(A_K) * B * (reset - u_equib)
    error = -C * (exp(A_K * tau) * (x_0 - x_equib + err_state) - err_state)
    return sum(error .^ 2)
end

function Sample_ball(n)
    v = randn(rng, Float64, (n))
    return v / norm(v)
end

function Sample_x0(n)
    return 2 * rand(rng, Float64, n) .- 1
    #return rand(rng, Float64, n)
end

function Sample_x0_mat(N, n)
    return 2 * rand(rng, Float64, (N, n)) .- 1
    #return rand(rng, Float64, (N, n))
end

function compute_Sigma(x_equib, n, p)
    Sigma = zeros((n + p), (n + p))
    Nsample = 5000
    for i in 1:Nsample
        x_0 = Sample_x0(n)
        Sigma += [(x_0-x_equib)*(x_0 - x_equib)' zeros(n, p); zeros(p, n) zeros(p, p)]
    end
    return Sigma / Nsample
end

function Sigma_calc(x_equib, n, p)
    ans = zeros(n + p, n + p)
    for i in 1:1000
        x0 = Sample_x0(n)
        ans += [(x0-x_equib)*(x0 - x_equib)' zeros(n, p); zeros(p, n) zeros(p, p)]
    end
    return ans / 1000
end


function grad_reset_estimate(tau, A_K, B, C, reset, x_equib, u_equib, N_reset, r_reset)
    m = size(reset, 1)
    n = size(x_equib, 1)
    ans_grad = zeros(m)
    for i in 1:N_reset
        x_0 = Sample_x0(n)
        v = Sample_ball(m) * sqrt(m)
        cost = error_tau(tau, A_K, B, C, reset + r_reset * v, x_0, x_equib, u_equib)
        base = error_tau(tau, A_K, B, C, reset, x_0, x_equib, u_equib)
        ans_grad += (cost - base) * v
    end
    return ans_grad ./ (r_reset * N_reset)
end

function grad_reset_estimate_normal(tau, A_K, B, C, reset, x_equib, u_equib, N_reset, r_reset)
    m = size(reset, 1)
    n = size(x_equib, 1)
    ans_grad = zeros(m)
    grads = zeros(m, N_reset)
    for i in 1:N_reset
        x_0 = Sample_x0(n)
        v = randn(rng, Float64, m)
        cost = error_tau(tau, A_K, B, C, reset + r_reset * v, x_0, x_equib, u_equib)
        base = error_tau(tau, A_K, B, C, reset, x_0, x_equib, u_equib)
        ans_grad += (cost - base) * v
        grads[:, i] = ans_grad ./ (r_reset * i)
    end
    ans_grad = ans_grad ./ ((r_reset^2) * N_reset)
    return ans_grad, grads
end

function grad_reset_estimate_history(tau, A_K, B, C, reset, x_equib, u_equib, N_reset, N_base, r_reset)
    m = size(reset, 1)
    n = size(x_equib, 1)
    ans_grad = zeros(m)
    grads = zeros(m, N_reset)
    base = 0
    for i in 1:N_base
        x_0 = Sample_x0(n)
        v = Sample_ball(m) * sqrt(m)
        base += error_tau(tau, A_K, B, C, reset + r_reset * v, x_0, x_equib, u_equib)
    end
    base = base / N_base
    #真のbaselineはg(m_0+rv)の期待値

    for i in 1:N_reset
        x_0 = Sample_x0(n)
        v = Sample_ball(m) * sqrt(m)
        cost = error_tau(tau, A_K, B, C, reset + r_reset * v, x_0, x_equib, u_equib)
        base = error_tau(tau, A_K, B, C, reset, x_0, x_equib, u_equib)

        ans_grad += (cost - base) * v
        grads[:, i] = ans_grad ./ (r_reset * i)
    end
    return grads
end

function grad_reset_estimate_two(tau, A_K, B, C, reset, x_equib, u_equib, N_reset, r_reset)
    m = size(reset, 1)
    n = size(x_equib, 1)
    ans_grad = zeros(m)
    grads = zeros(m, N_reset)

    for i in 1:N_reset
        x_0 = Sample_x0(n)
        v = Sample_ball(m) * sqrt(m)
        cost1 = error_tau(tau, A_K, B, C, reset + r_reset * v, x_0, x_equib, u_equib)
        cost2 = error_tau(tau, A_K, B, C, reset - r_reset * v, x_0, x_equib, u_equib)

        ans_grad += (cost1 - cost2) * v
        grads[:, i] = ans_grad ./ (2 * r_reset * i)
    end
    return grads
end

function grad_reset_estimate_samplebase(tau, A_K, B, C, reset, x_equib, u_equib, N_reset, N_base, r_reset)
    m = size(reset, 1)
    n = size(x_equib, 1)
    ans_grad = zeros(m)
    grads = zeros(m, N_reset)
    base = 0
    for i in 1:N_base
        x_0 = Sample_x0(n)
        v = Sample_ball(m) * sqrt(m)
        base += error_tau(tau, A_K, B, C, reset + r_reset * v, x_0, x_equib, u_equib)
    end
    base = base / N_base
    #真のbaselineはg(m_0+rv)の期待値

    for i in 1:N_reset
        x_0 = Sample_x0(n)
        v = Sample_ball(m) * sqrt(m)
        cost = error_tau(tau, A_K, B, C, reset + r_reset * v, x_0, x_equib, u_equib)

        ans_grad += (cost - base) * v
        grads[:, i] = ans_grad ./ (r_reset * i)
    end
    return grads
end

function grad_reset(F_K, G, K_M, reset, u_equib)
    sub_mat = inv(F_K) * G * K_M
    err_state = sub_mat * (reset - u_equib)
    error = H[1:p, :] * err_state

    return 2 * sub_mat' * H[1:p, :]' * error
end

function estimate_reset(tau, A_K, B, C, reset, x_equib, u_equib, N_reset, r_reset, eps_reset)
    #P制御の無限次元経過後の誤差を最小化する
    f_list = []
    reset_list = []
    push!(reset_list, reset)
    n = size(x_equib, 1)
    p = size(reset, 1)
    rho = 0.95
    c = 0.1
    cnt = 0
    x_0 = Sample_x0(n)
    val = error_tau(tau, A_K, B, C, reset, x_0, x_equib, u_equib)
    Sigma = Sigma_calc(x_equib, n, p)
    println("目的関数: ", val)
    push!(f_list, val)
    while true
        alpha = 1
        est_grad = grad_reset_estimate(tau, A_K, B, C, reset, x_equib, u_equib, N_reset, r_reset)
        reset_next = reset - alpha * est_grad
        val = error_tau(tau, A_K, B, C, reset, x_0, x_equib, u_equib)
        while (error_tau(tau, A_K, B, C, reset_next, x_0, x_equib, u_equib) >= val + c * alpha * est_grad' * est_grad && alpha >= 1e-20)
            alpha *= rho
            reset_next = reset - alpha * est_grad
        end
        cnt += 1
        val_next = error_tau(tau, A_K, B, C, reset_next, x_0, x_equib, u_equib)
        #println(f_val)
        if (cnt % 1000 == 0)
            println(cnt)
            println(val_next)
            println("勾配の推定", est_grad)
            println("勾配", 2 * ((C * inv(A_K) * B * K_M)' * (C * inv(A_K) * B * K_M)) * (reset - u_equib))
        end
        push!(reset_list, reset)
        push!(f_list, val_next)
        if ((val_next < eps_reset) || alpha <= 1e-20)
            return reset_list, f_list
        end
        reset = reset_next
        val = val_next
    end
end

function estimate_reset_ConstStep(tau, A_K, B, C, reset, x_equib, u_equib, N_reset, r_reset, N_GD, eta)
    #P制御の無限次元経過後の誤差を最小化する
    f_list = []
    reset_list = []
    push!(reset_list, reset)
    n = size(x_equib, 1)
    p = size(reset, 1)
    rho = 0.95
    c = 0.1
    cnt = 0
    x_0 = Sample_x0(n)
    err_state = inv(A_K) * B * (reset - u_equib)
    val = sum((C * err_state) .^ 2)
    Sigma = Sigma_calc(x_equib, n, p)
    println("目的関数: ", val)
    push!(f_list, val)
    while cnt < N_GD
        est_grad = grad_reset_estimate(tau, A_K, B, C, reset, x_equib, u_equib, N_reset, r_reset)
        reset = reset - eta * est_grad
        cnt += 1
        err_state = inv(A_K) * B * (reset - u_equib)
        val_next = sum((C * err_state) .^ 2)
        #println(f_val)
        if (cnt % 1000 == 0)
            println(cnt)
            println(val_next)
            #println("勾配の推定", est_grad)
            #println("勾配", 2 * ((C * inv(A_K) * B * K_M)' * (C * inv(A_K) * B * K_M)) * (reset - u_equib))
        end
        push!(reset_list, reset)
        push!(f_list, val_next)
        val = val_next
    end
    return reset_list, f_list
end

function grad_obj(F_K, G, H, K_P, K_I, K_D, C_aug, Sigma)
    p = size(K_P, 1)
    K_M = inv(I(p) + K_D * H[1:p, 1:n] * G[1:n, :])
    K_Pbar = K_M * K_P
    K_Ibar = K_M * K_I
    K_Dbar = K_M * K_D
    K_bar = [K_Pbar K_Ibar K_Dbar]
    X = lyap(F_K, Sigma)
    Y = lyap(F_K', C_aug)
    Z = -2 * G' * Y * X * H'
    KmZ = K_M' * Z
    grad_Kp = KmZ[:, 1:p]
    grad_Ki = KmZ[:, (p+1):2*p]
    grad_Kd = KmZ[:, (2*p+1):end] - KmZ * K_bar' * K_M' * (H[1:p, 1:n] * G[1:n, :])'
    grad_Kp = (grad_Kp + grad_Kp') / 2
    grad_Ki = (grad_Ki + grad_Ki') / 2
    grad_Kd = (grad_Kd + grad_Kd') / 2
    grad_P = diag(grad_Kp)
    grad_I = diag(grad_Ki)
    grad_D = diag(grad_Kd)
    return grad_P, grad_I, grad_D
end

function obj_trunc(F, G, H, K_P, K_I, K_D, x_0, x_equib, u_equib, reset, tau, n, p, h)
    F_K = computeF_K(F, G, H, K_P, K_I, K_D)

    Nstep = Int(tau / h)

    x_aug_init = [x_0 - x_equib; zeros(p)]
    C = H[1:p, 1:n]
    error = C * x_aug_init[1:n]

    K_M = inv(I(p) + K_D * H[1:p, 1:n] * G[1:n, :])

    equib_vec = inv(F_K) * G * K_M * (reset - u_equib)


    obj = sum(error .^ 2) / 3
    @simd for j in 1:(Nstep-1)
        x_aug = exp(F_K * j * h) * (x_aug_init + equib_vec) - equib_vec
        error = C * x_aug[1:n]
        if j % 2 == 1
            obj += 4 * sum(error .^ 2) / 3
        else
            obj += 2 * sum(error .^ 2) / 3
        end
    end
    x_aug = exp(F_K * tau) * (x_aug_init + equib_vec) - equib_vec
    error = C * x_aug[1:n]
    obj += sum(error .^ 2) / 3
    obj *= h
    return obj
end

function obj_trunc_FK(F_K, equib_vec, x_aug_init, C, h, n, tau)
    Nstep = Int(tau / h)
    error = -C * x_aug_init[1:n]
    obj = sum(error .^ 2) / 3
    @simd for j in 1:(Nstep-1)
        x_aug = exp(F_K * j * h) * (x_aug_init + equib_vec) - equib_vec
        error = -C * x_aug[1:n]
        if j % 2 == 1
            obj += 4 * sum(error .^ 2) / 3
        else
            obj += 2 * sum(error .^ 2) / 3
        end
    end
    x_aug = exp(F_K * tau) * (x_aug_init + equib_vec) - equib_vec
    error = -C * x_aug[1:n]
    obj += sum(error .^ 2) / 3
    obj *= h
    return obj
end

function Compute_ErrBar(F_K, equib_vec, x_aug_init, C, h, h_samp, n, p, s, D)
    errors_prior = zeros(D * p) #D個errorのサンプルを撮る
    error = -C * x_aug_init[1:n]
    errors_prior[1:p] = error
    #エラーを縦に並べたベクトルを作る（errors_prior)
    for j in 1:(D-1)
        x_aug = exp(F_K * j * h_samp) * (x_aug_init + equib_vec) - equib_vec
        error = -C * x_aug[1:n]
        errors_prior[(j*p+1):((j+1)*p)] = error
    end
    obj_s = obj_trunc_FK(F_K, equib_vec, x_aug_init, C, h, n, s)#時刻sでの目的関数 

    errors_prior = kron(errors_prior, errors_prior)

    x_aug = exp(F_K * s) * (x_aug_init + equib_vec) - equib_vec

    errors_post = zeros(D * p)
    errors_post[1:p] = -C * x_aug[1:n]
    for j in 1:(D-1)
        x_aug = exp(F_K * (j * h_samp + s)) * (x_aug_init + equib_vec) - equib_vec
        error = -C * x_aug[1:n]
        errors_post[(j*p+1):((j+1)*p)] = error
    end

    errors_post = kron(errors_post, errors_post)
    errors = errors_prior - errors_post
    return errors, obj_s
end

function baseline_grad_obj(F_K, G, K_M, x_equib, u_equib, reset, h, h_samp, s, D, N_base)
    #何回かサンプルしてex(0)' * P *ex(0) =obj_trunc(ex(0))を満たすようなPを求めてあげたい．
    n = size(x_equib, 1)
    #N_base = 100 #変えるかも. D*p個or (n+1)*n/2個
    equib_vec = inv(F_K) * G * K_M * (reset - u_equib)
    E_mat = zeros(N_base, (D * p)^2) #Pのベクトル化が列に並ぶ
    vals = zeros(N_base) #各サンプルごとの目的関数ち
    for i in 1:N_base
        x_0 = Sample_x0(n)
        x_aug_init = [x_0 - x_equib; zeros(p)]
        errors, obj_s = Compute_ErrBar(F_K, equib_vec, x_aug_init, C, h, h_samp, n, p, s, D)
        E_mat[i, :] = errors
        vals[i] = obj_s
    end
    vecP = E_mat \ vals
    return vecP
end

function baseline_grad_obj_trial(F_K, G, K_M, x_equib, u_equib, reset, h, h_samp, s, D, N_base)
    #何回かサンプルしてex(0)' * P *ex(0) =obj_trunc(ex(0))を満たすようなPを求めてあげたい．
    n = size(x_equib, 1)
    #N_base = 100 #変えるかも. D*p個or (n+1)*n/2個
    equib_vec = inv(F_K) * G * K_M * (reset - u_equib)
    E_mat = zeros(N_base, (D * p)^2) #Pのベクトル化が列に並ぶ
    vals = zeros(N_base) #各サンプルごとの目的関数ち
    for i in 1:N_base
        x_0 = Sample_x0(n)
        x_aug_init = [x_0 - x_equib; randn(p)]
        errors, obj_s = Compute_ErrBar(F_K, equib_vec, x_aug_init, C, h, h_samp, n, p, s, D)
        E_mat[i, :] = errors
        vals[i] = obj_s
    end
    vecP = E_mat \ vals
    return vecP
end

function obj_trunc_Base(F, G, H, K_P, K_I, K_D, x_0, x_equib, u_equib, reset, tau, n, p, h, h_samp, D)
    #ある初期点でのベースライン計算に使う軌道の情報と目的関数を求める．

    #軌道からの数サンプルから初期点を計算する．
    K_M = inv(I(p) + K_D * H[1:p, 1:n] * G[1:n, :])
    K_Pbar = K_M * K_P
    K_Ibar = K_M * K_I
    K_Dbar = K_M * K_D
    F_K = F - G * [K_Pbar K_Ibar K_Dbar] * H

    x_aug_init = [x_0 - x_equib; zeros(p)]
    C = H[1:p, 1:n]
    error = -C * x_aug_init[1:n]

    equib_vec = inv(F_K) * G * K_M * (reset - u_equib)
    errors_base1 = zeros(D * p)#時刻0からのサンプル
    errors_base1[1:p] = error
    @simd for j in 1:(D-1)
        x_aug = exp(F_K * j * h_samp) * (x_aug_init + equib_vec) - equib_vec
        error = -C * x_aug[1:n]
        errors_base1[p*j+1:p*(j+1)] = error
    end
    errors_base1 = kron(errors_base1, errors_base1)

    x_aug = exp(F_K * tau) * (x_aug_init + equib_vec) - equib_vec
    error = -C * x_aug[1:n]
    errors_base2 = zeros(D * p)#時刻tauからのサンプル
    errors_base2[1:p] = error

    @simd for j in 1:(D-1)
        x_aug = exp(F_K * (tau + j * h_samp)) * (x_aug_init + equib_vec) - equib_vec
        error = -C * x_aug[1:n]
        errors_base2[p*j+1:p*(j+1)] = error
    end

    errors_base2 = kron(errors_base2, errors_base2)
    #目的関数の計算
    obj = obj_trunc_FK(F_K, equib_vec, x_aug_init, C, h, n, tau)

    return obj, errors_base1, errors_base2
end


function grad_obj_est_Base(K_P, K_I, K_D, F, G, H, N, x_equib, u_equib, reset, tau, r, h)
    """
    モデルフリーな勾配の推定ほう
    ベクトル形式の勾配を返す．
    """
    p = size(K_P, 1)
    n = size(F, 1) - p

    grad_Ps = zeros(p, N)
    grad_Is = zeros(p, N)
    grad_Ds = zeros(p, N)

    grad_P = zeros(p)
    grad_I = zeros(p)
    grad_D = zeros(p)

    zp_s = randn(rng, Float64, (N, 3 * p))
    x0_Mat = Sample_x0_mat(3 * N, n)
    D = 2 * (n + p - 1) + 20
    tau_prime = 1
    h_samp = tau_prime / D
    s = 20
    N_base = Int((n + p) * (n + p + 1) / 2) #ベースライン計算に何回反復するか
    K_M = inv(I(p) + K_D * H[1:p, 1:n] * G[1:n, :])
    K_Pbar = K_M * K_P
    K_Ibar = K_M * K_I
    K_Dbar = K_M * K_D
    F_K = F - G * [K_Pbar K_Ibar K_Dbar] * H
    base_vecP = baseline_grad_obj_trial(F_K, G, K_M, x_equib, u_equib, reset, h, h_samp, s, D, N_base)

    @simd for i in 1:N
        #zp = zp_s[i, :]

        zp = zp_s[i, :]
        zp = zp / norm(zp)
        zp = zp * sqrt(3 * p)
        U_P = zp[1:p]
        U_I = zp[(p+1):(2*p)]
        U_D = zp[(2*p+1):(3*p)]
        x_0 = Sample_x0(n)
        #ある初期点からの摂動を入れたゲインでの目的関数の計算，　
        cost, errors_base1, errors_base2 = obj_trunc_Base(F, G, H, K_P + r * diagm(U_P), K_I + r * diagm(U_I), K_D + r * diagm(U_D), x_0, x_equib, u_equib, reset, tau, n, p, h, h_samp, D)
        base = (errors_base1 - errors_base2)' * base_vecP

        grad_P += (cost - base) .* U_P
        grad_I += (cost - base) .* U_I
        grad_D += (cost - base) .* U_D


        grad_Ps[:, i] = grad_P ./ (r * i)
        grad_Is[:, i] = grad_I ./ (r * i)
        grad_Ds[:, i] = grad_D ./ (r * i)
    end
    grad_P = grad_P ./ (r * N)
    grad_I = grad_I ./ (r * N)
    grad_D = grad_D ./ (r * N)

    return grad_P, grad_I, grad_D, grad_Ps, grad_Is, grad_Ds
end

function grad_obj_est_WithoutBase(K_P, K_I, K_D, F, G, H, N, x_equib, u_equib, reset, tau, r, h)
    """
    モデルフリーな勾配の推定ほう
    ベクトル形式の勾配を返す．
    """
    p = size(K_P, 1)
    n = size(F, 1) - p

    grad_Ps = zeros(p, N)
    grad_Is = zeros(p, N)
    grad_Ds = zeros(p, N)

    grad_P = zeros(p)
    grad_I = zeros(p)
    grad_D = zeros(p)

    zp_s = randn(rng, Float64, (N, 3 * p))
    #x0_Mat = Sample_x0_mat(3 * N, n)


    @simd for i in 1:N
        #zp = zp_s[i, :]

        zp = zp_s[i, :]
        zp = zp / norm(zp)
        zp = zp * sqrt(3 * p)
        U_P = zp[1:p]
        U_I = zp[(p+1):(2*p)]
        U_D = zp[(2*p+1):(3*p)]
        x_0 = Sample_x0(n)
        #ある初期点からの摂動を入れたゲインでの目的関数の計算，　
        cost = obj_trunc(F, G, H, K_P + r * diagm(U_P), K_I + r * diagm(U_I), K_D + r * diagm(U_D), x_0, x_equib, u_equib, reset, tau, n, p, h)

        grad_P += cost * U_P
        grad_I += cost * U_I
        grad_D += cost * U_D

        grad_Ps[:, i] = grad_P / (r * i)
        grad_Is[:, i] = grad_I / (r * i)
        grad_Ds[:, i] = grad_D / (r * i)
    end
    grad_P = grad_P ./ (r * N)
    grad_I = grad_I ./ (r * N)
    grad_D = grad_D ./ (r * N)

    return grad_P, grad_I, grad_D, grad_Ps, grad_Is, grad_Ds
end

function Projection_diagnal_interval(K, eps_interval, M_interval)
    p = size(K, 1)
    for i in 1:p
        if K[i, i] < eps_interval
            K[i, i] = eps_interval
        elseif K[i, i] > M_interval
            K[i, i] = M_interval
        end
    end
    return K
end

function ProjectGradient_Gain_Conststep(tau, eps_interval, M_interval, K_P, K_I, K_D, F, G, H, C_aug, N_sample, x_equib, u_equib, reset, r, h, N_GD, eta, epsilon)
    #P制御の無限次元経過後の誤差を最小化する
    f_list = []
    Kp_list = []
    Ki_list = []
    Kd_list = []
    push!(Kp_list, K_P)
    push!(Ki_list, K_I)
    push!(Kd_list, K_D)
    n = size(x_equib, 1)
    p = size(K_P, 1)

    cnt = 0
    Sigma = Sigma_calc(x_equib, n, p)
    val = ObjectiveFunction(F, G, H, C_aug, Sigma, K_P, K_I, K_D)
    println("目的関数: ", val)
    push!(f_list, val)

    while cnt < N_GD
        grad_P, grad_I, grad_D, _, _, _ = grad_obj_est_Base(K_P, K_I, K_D, F, G, H, N_sample, x_equib, u_equib, reset, tau, r, h)
        K_P_next = K_P - eta * diagm(grad_P)
        K_I_next = K_I - eta * diagm(grad_I)
        K_D_next = K_D - eta * diagm(grad_D)
        K_P_next = Projection_diagnal_interval(K_P_next, eps_interval, M_interval)
        K_I_next = Projection_diagnal_interval(K_I_next, eps_interval, M_interval)
        K_D_next = Projection_diagnal_interval(K_D_next, eps_interval, M_interval)
        cnt += 1
        val = ObjectiveFunction(F, G, H, C_aug, Sigma, K_P_next, K_I_next, K_D_next)
        #射影する
        #println(f_val)
        difference = sqrt(sum((K_P_next - K_P) .^ 2) + sum((K_I_next - K_I) .^ 2) + sum((K_D_next - K_D) .^ 2))
        if (difference < epsilon * eta)
            push!(Kp_list, K_P)
            push!(Ki_list, K_I)
            push!(Kd_list, K_D)
            push!(f_list, val)
            return Kp_list, Ki_list, Kd_list, f_list
        end
        if (cnt % 100 == 0)
            println(cnt)
            println(val)
            #println("勾配の推定", est_grad)
            #println("勾配", 2 * ((C * inv(A_K) * B * K_M)' * (C * inv(A_K) * B * K_M)) * (reset - u_equib))
        end
        K_P = K_P_next
        K_I = K_I_next
        K_D = K_D_next
        push!(Kp_list, K_P)
        push!(Ki_list, K_I)
        push!(Kd_list, K_D)

        push!(f_list, val)

    end
    return Kp_list, Ki_list, Kd_list, f_list
end

function ProjectGradient_Gain_Conststep_SemoDefinite(tau, eps_interval, M_interval, K_P, K_I, K_D, F, G, H, C_aug, N_sample, x_equib, u_equib, reset, r, h, N_GD, eta, epsilon)
    #P制御の無限次元経過後の誤差を最小化する
    f_list = []
    Kp_list = []
    Ki_list = []
    Kd_list = []
    push!(Kp_list, K_P)
    push!(Ki_list, K_I)
    push!(Kd_list, K_D)
    n = size(x_equib, 1)
    p = size(K_P, 1)

    cnt = 0
    Sigma = Sigma_calc(x_equib, n, p)
    val = ObjectiveFunction(F, G, H, C_aug, Sigma, K_P, K_I, K_D)
    println("目的関数: ", val)
    push!(f_list, val)

    while cnt < N_GD
        grad_P, grad_I, grad_D, _, _, _ = grad_obj_est_Base(K_P, K_I, K_D, F, G, H, N_sample, x_equib, u_equib, reset, tau, r, h)
        K_P_next = K_P - eta * diagm(grad_P)
        K_I_next = K_I - eta * diagm(grad_I)
        K_D_next = K_D - eta * diagm(grad_D)
        K_P_next = Projection_diagnal_interval(K_P_next, eps_interval, M_interval)
        K_I_next = Projection_diagnal_interval(K_I_next, 0, M_interval)
        K_D_next = Projection_diagnal_interval(K_D_next, 0, M_interval)
        cnt += 1
        val = ObjectiveFunction(F, G, H, C_aug, Sigma, K_P_next, K_I_next, K_D_next)
        #射影する
        #println(f_val)
        difference = sqrt(sum((K_P_next - K_P) .^ 2) + sum((K_I_next - K_I) .^ 2) + sum((K_D_next - K_D) .^ 2))
        if (difference < epsilon * eta)
            push!(Kp_list, K_P)
            push!(Ki_list, K_I)
            push!(Kd_list, K_D)
            push!(f_list, val)
            return Kp_list, Ki_list, Kd_list, f_list
        end
        if (cnt % 100 == 0)
            println(cnt)
            println(val)
            #println("勾配の推定", est_grad)
            #println("勾配", 2 * ((C * inv(A_K) * B * K_M)' * (C * inv(A_K) * B * K_M)) * (reset - u_equib))
        end
        K_P = K_P_next
        K_I = K_I_next
        K_D = K_D_next
        push!(Kp_list, K_P)
        push!(Ki_list, K_I)
        push!(Kd_list, K_D)

        push!(f_list, val)

    end
    return Kp_list, Ki_list, Kd_list, f_list
end



function ProjectGradient_Gain_Conststep_ModelBased(eps_interval, M_interval, K_P, K_I, K_D, F, G, H, C_aug, x_equib, N_GD, eta, epsilon)
    #P制御の無限次元経過後の誤差を最小化する
    f_list = []
    Kp_list = []
    Ki_list = []
    Kd_list = []
    push!(Kp_list, K_P)
    push!(Ki_list, K_I)
    push!(Kd_list, K_D)
    n = size(x_equib, 1)
    p = size(K_P, 1)

    cnt = 0
    Sigma = Sigma_calc(x_equib, n, p)
    val = ObjectiveFunction(F, G, H, C_aug, Sigma, K_P, K_I, K_D)
    println("目的関数: ", val)
    push!(f_list, val)

    while cnt < N_GD
        F_K = computeF_K(F, G, H, K_P, K_I, K_D)
        grad_P, grad_I, grad_D = grad_obj(F_K, G, H, K_P, K_I, K_D, C_aug, Sigma)
        K_P_next = K_P - eta * diagm(grad_P)
        K_I_next = K_I - eta * diagm(grad_I)
        K_D_next = K_D - eta * diagm(grad_D)
        K_P_next = Projection_diagnal_interval(K_P_next, eps_interval, M_interval)
        K_I_next = Projection_diagnal_interval(K_I_next, eps_interval, M_interval)
        K_D_next = Projection_diagnal_interval(K_D_next, eps_interval, M_interval)
        cnt += 1
        val = ObjectiveFunction(F, G, H, C_aug, Sigma, K_P_next, K_I_next, K_D_next)
        #射影する
        #println(f_val)
        difference = sqrt(sum((K_P_next - K_P) .^ 2) + sum((K_I_next - K_I) .^ 2) + sum((K_D_next - K_D) .^ 2))
        if (difference < epsilon * eta)
            push!(Kp_list, K_P)
            push!(Ki_list, K_I)
            push!(Kd_list, K_D)
            push!(f_list, val)
            println(cnt)
            println(val)
            return Kp_list, Ki_list, Kd_list, f_list
        end
        if (cnt % 1000 == 0)
            println(cnt)
            println(val)
            #println("勾配の推定", est_grad)
            #println("勾配", 2 * ((C * inv(A_K) * B * K_M)' * (C * inv(A_K) * B * K_M)) * (reset - u_equib))
        end
        K_P = K_P_next
        K_I = K_I_next
        K_D = K_D_next
        push!(Kp_list, K_P)
        push!(Ki_list, K_I)
        push!(Kd_list, K_D)

        push!(f_list, val)

    end
    return Kp_list, Ki_list, Kd_list, f_list
end

function ProjectGradient_Gain_Conststep_ModelBased_Semidefinite(eps_interval, M_interval, K_P, K_I, K_D, F, G, H, C_aug, x_equib, N_GD, eta, epsilon)
    #P制御の無限次元経過後の誤差を最小化する
    f_list = []
    Kp_list = []
    Ki_list = []
    Kd_list = []
    push!(Kp_list, K_P)
    push!(Ki_list, K_I)
    push!(Kd_list, K_D)
    n = size(x_equib, 1)
    p = size(K_P, 1)

    cnt = 0
    Sigma = Sigma_calc(x_equib, n, p)
    val = ObjectiveFunction(F, G, H, C_aug, Sigma, K_P, K_I, K_D)
    println("目的関数: ", val)
    push!(f_list, val)

    while cnt < N_GD
        F_K = computeF_K(F, G, H, K_P, K_I, K_D)
        grad_P, grad_I, grad_D = grad_obj(F_K, G, H, K_P, K_I, K_D, C_aug, Sigma)
        K_P_next = K_P - eta * diagm(grad_P)
        K_I_next = K_I - eta * diagm(grad_I)
        K_D_next = K_D - eta * diagm(grad_D)
        K_P_next = Projection_diagnal_interval(K_P_next, eps_interval, M_interval)
        K_I_next = Projection_diagnal_interval(K_I_next, eps_interval, M_interval)
        K_D_next = Projection_diagnal_interval(K_D_next, 0, M_interval)
        cnt += 1
        val = ObjectiveFunction(F, G, H, C_aug, Sigma, K_P_next, K_I_next, K_D_next)
        #射影する
        #println(f_val)
        difference = sqrt(sum((K_P_next - K_P) .^ 2) + sum((K_I_next - K_I) .^ 2) + sum((K_D_next - K_D) .^ 2))
        if (difference < epsilon * eta)
            push!(Kp_list, K_P)
            push!(Ki_list, K_I)
            push!(Kd_list, K_D)
            push!(f_list, val)
            println(cnt)
            println(val)
            return Kp_list, Ki_list, Kd_list, f_list
        end
        if (cnt % 1000 == 0)
            println(cnt)
            println(val)
            #println("勾配の推定", est_grad)
            #println("勾配", 2 * ((C * inv(A_K) * B * K_M)' * (C * inv(A_K) * B * K_M)) * (reset - u_equib))
        end
        K_P = K_P_next
        K_I = K_I_next
        K_D = K_D_next
        push!(Kp_list, K_P)
        push!(Ki_list, K_I)
        push!(Kd_list, K_D)

        push!(f_list, val)

    end
    return Kp_list, Ki_list, Kd_list, f_list
end