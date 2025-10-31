function error_tau_nonlinear(tau, F, G, nablaH, K_P, K_I, K_D, x_0, reset, y_ans, h, n)
    p = size(y_ans, 1)
    if p == 1
        _, y_s, _, _ = Orbit_nonlinear_p1_P(F, G, nablaH, K_P, K_I, K_D, x_0, reset, y_ans, h, tau, n)
        error = (y_s[end] - y_ans[1])
        return sum(error .^ 2)
    else
        _, y_s, _, _ = Orbit_nonlinear(F, G, nablaH, K_P, K_I, K_D, x_0, reset, y_ans, h, tau, n, p)
        error = (y_s[end] - y_ans)
        return sum(error .^ 2)
    end
end

function grad_reset_estimate_nonlinear(tau, F, G, nablaH, K_P, K_I, K_D, reset, y_ans, h, n, N_reset, r_reset)
    m = size(reset, 1)
    ans_grad = zeros(m)
    for i in 1:N_reset
        x_0 = Sample_x0(n)
        v = Sample_ball(m) * sqrt(m)
        cost = error_tau_nonlinear(tau, F, G, nablaH, K_P, K_I, K_D, x_0, reset + r_reset * v, y_ans, h, n)
        x_0 = Sample_x0(n)
        base = error_tau_nonlinear(tau, F, G, nablaH, K_P, K_I, K_D, x_0, reset, y_ans, h, n)
        ans_grad += (cost - base) * v
    end
    return ans_grad ./ (r_reset * N_reset)
end


function estimate_reset_nonlinear(tau, F, G, nablaH, K_P, K_I, K_D, reset, y_ans, h, n, N_reset, r_reset, eps_reset)
    #P制御の無限次元経過後の誤差を最小化する
    f_list = []
    reset_list = []
    push!(reset_list, reset)
    p = size(reset, 1)
    rho = 0.95
    c = 0.1
    cnt = 0
    x_0 = Sample_x0(n)
    val = error_tau_nonlinear(tau, F, G, nablaH, K_P, K_I, K_D, x_0, reset, y_ans, h, n)
    println("目的関数: ", val)
    push!(f_list, val)
    while true
        alpha = 1
        est_grad = grad_reset_estimate_nonlinear(tau, F, G, nablaH, K_P, K_I, K_D, reset, y_ans, h, n, N_reset, r_reset)
        reset_next = reset - alpha * est_grad
        val = error_tau_nonlinear(tau, F, G, nablaH, K_P, K_I, K_D, x_0, reset, y_ans, h, n)

        while (error_tau_nonlinear(tau, F, G, nablaH, K_P, K_I, K_D, x_0, reset_next, y_ans, h, n) >= val + c * alpha * est_grad' * est_grad && alpha >= 1e-20)
            alpha *= rho
            reset_next = reset - alpha * est_grad
        end
        cnt += 1
        val_next = error_tau_nonlinear(tau, F, G, nablaH, K_P, K_I, K_D, x_0, reset_next, y_ans, h, n)
        #println(f_val)
        if (cnt % 50 == 0)
            println(cnt)
            println(val_next)
            println("勾配の推定", est_grad)
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