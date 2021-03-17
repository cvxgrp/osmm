import numpy as np
# from hessian_est import compute_hessian_eig_decomp


class CurvatureUpdate:
    def __init__(self, problem_instance):
        self.problem_instance = problem_instance

    def low_rank_quasi_Newton_update(self, U_k, xcur, xprev, grad_cur, grad_prev, tol_1=1e-10, tol_2=1e-2, ep=1e-15):
        s = xcur - xprev
        y = grad_cur - grad_prev
        y_abs_too_small_idxes = np.where(np.abs(y) <= ep)[0]
        y[y_abs_too_small_idxes] = 0
        yTs = y.T.dot(s)
        _, r = U_k.shape
        r1 = r
        v = U_k.T.dot(s)
        if yTs < max(tol_1, np.linalg.norm(y) * np.linalg.norm(s) * tol_2):
            if np.linalg.norm(s) < tol_2 * np.linalg.norm(xprev) or np.linalg.norm(v) < tol_1:
                # print("U_k not updated", np.linalg.norm(s), tol_2 * np.linalg.norm(xprev), np.linalg.norm(v))
                return U_k
            r1 = -1
        while r1 > 0 and yTs - v[0:r1].T.dot(v[0:r1]) < np.linalg.norm(y - U_k[:, 0:r1].dot(v[0:r1])) \
                * np.linalg.norm(s) * tol_2:
            r1 -= 1
        if r1 >= 0:
            v1 = v[0:r1]
            U1 = U_k[:, 0:r1]
            alpha_k = np.sqrt(yTs - v1.T.dot(v1))
            P = np.fliplr(np.eye(r1 + 1))
            tmp = np.eye(r1 + 1)
            tmp[0, 0] = 1.0 / alpha_k
            tmp[1:r1 + 1, 0] = - v1 / alpha_k
            _, R1_tilde = np.linalg.qr(np.transpose(P.dot(tmp)))
            R1 = P.dot(R1_tilde.T.dot(P))
            new_U1 = np.concatenate([y.reshape((self.problem_instance.n, 1)), U1], axis=1).dot(R1)  # n by r1+1
        if r1 == r:
            U_k_plus_one = new_U1[:, 0:r]
        else:
            r2 = r - max(0, r1)
            v2 = v[r - r2:r]
            U2 = U_k[:, r - r2:r]
            basis, _, _ = np.linalg.svd(v2.reshape((r2, 1)))
            Q2 = np.array(basis)
            Q2[:, r2 - 1] = basis[:, 0]
            Q2[:, 0:r2 - 1] = basis[:, 1::]
            U2Q2 = U2.dot(Q2)
            new_U2 = U2Q2[:, 0:r2 - 1]  # n by r2-1
            if r1 >= 0:
                U_k_plus_one = np.concatenate([new_U1, new_U2], axis=1)
            else:
                U_k_plus_one = U_k
                U_k_plus_one[:, 0:r - 1] = new_U2
                U_k_plus_one[:, r - 1] = np.zeros(self.problem_instance.n)
        return U_k_plus_one

    def BFGS_update(self, U_k, xcur, xprev, grad_cur, grad_prev, tol=1e-10):
        s = xcur - xprev
        y = grad_cur - grad_prev
        yTs = y.T.dot(s)
        H_k = U_k.dot(U_k.T)
        if yTs > tol:
            Hs = H_k.dot(s)
            H_k_plus_ones = H_k + np.outer(y, y) / yTs - np.outer(Hs, Hs) / s.T.dot(Hs)
            u_vec, s_arr, _ = np.linalg.svd(H_k_plus_ones)
            s_arr = np.sqrt(np.maximum(0, s_arr))
            U_k_plus_one = u_vec.dot(np.diag(s_arr))
        else:
            U_k_plus_one = U_k
        return U_k_plus_one

    def LBFGS_update(self, U_k, xcur, xprev, grad_cur, grad_prev, pieces_num, Y_lbfgs, S_lbfgs, lbfgs_update_round_idx,
                     tol=1e-10):
        s = xcur - xprev
        y = grad_cur - grad_prev
        yTs = y.T.dot(s)
        if yTs > tol:
            delta_lbfgs = np.sum(np.square(y)) / yTs
            Y_lbfgs[:, (lbfgs_update_round_idx - 1) % pieces_num] = y
            S_lbfgs[:, (lbfgs_update_round_idx - 1) % pieces_num] = s
            tmp_pieces_num = min(lbfgs_update_round_idx, pieces_num)
            LBFGS_delta_S_Y = np.zeros((self.problem_instance.n, tmp_pieces_num * 2))
            LBFGS_delta_S_Y[:, 0:tmp_pieces_num] = delta_lbfgs * S_lbfgs[:, 0:tmp_pieces_num]
            LBFGS_delta_S_Y[:, tmp_pieces_num:tmp_pieces_num * 2] = Y_lbfgs[:, 0:tmp_pieces_num]
            STY = S_lbfgs[:, 0:tmp_pieces_num].T.dot(Y_lbfgs[:, 0:tmp_pieces_num])
            STS = S_lbfgs[:, 0:tmp_pieces_num].T.dot(S_lbfgs[:, 0:tmp_pieces_num])
            LBFGS_middle = np.zeros((tmp_pieces_num * 2, tmp_pieces_num * 2))
            LBFGS_middle[0:tmp_pieces_num, 0:tmp_pieces_num] = delta_lbfgs * STS
            LBFGS_middle[tmp_pieces_num:2 * tmp_pieces_num, tmp_pieces_num:2 * tmp_pieces_num] = np.diag(-np.diag(STY))
            LBFGS_middle[0:tmp_pieces_num, tmp_pieces_num:2 * tmp_pieces_num] = np.tril(STY, -1)
            LBFGS_middle[tmp_pieces_num:2 * tmp_pieces_num, 0:tmp_pieces_num] = np.tril(STY, -1).T
            mid_inv = np.linalg.inv(LBFGS_middle)
            Q_fac, R_fac = np.linalg.qr(LBFGS_delta_S_Y)
            eigen_vals, eigen_vecs = np.linalg.eig(R_fac.dot(mid_inv.dot(R_fac.T)))
            sqrt_eigen_vals = np.sqrt(delta_lbfgs - eigen_vals)
            U_k[:, 0:tmp_pieces_num * 2] = Q_fac.dot(eigen_vecs.dot(np.diag(sqrt_eigen_vals)))
            # self.H_lbfgs = self.delta_lbfgs * np.eye(self.n) - LBFGS_delta_S_Y.dot(mid_inv.dot(LBFGS_delta_S_Y.T))
            lbfgs_update_round_idx += 1
        return U_k, lbfgs_update_round_idx

    def exact_hess_update(self, x):
        W_minibatch = None
        H = self.problem_instance.f_hess(x)
        u_vec, s_arr, _ = np.linalg.svd(H)
        s_arr = np.sqrt(s_arr)
        U = u_vec.dot(np.diag(s_arr))
        return U

    def trace_hess_update(self, x):
        W_minibatch = None
        H = self.problem_instance.f_hess(x)
        diag_H = np.trace(H) / self.problem_instance.n * np.ones(self.problem_instance.n)
        return diag_H

    def diag_hess_update(self, x):
        W_minibatch = None
        H = self.problem_instance.f_hess(x)
        diag_H = np.diag(H)
        return diag_H

    def low_rank_diag_update(self, x, H_rank):
        W_minibatch = None
        if self.problem_instance.n > 10000:
            print("Fast eigen-decomposition for n >= 10000 is commented out.")
            # if H_rank == n:
            #     print("ERROR: Rank must be less than n for fast eigen-decomposition.")
            # if W_minibatch is None:
            #     s_arr, eigen_vec = compute_hessian_eig_decomp(x, W, f_torch, H_rank + 1)
            # else:
            #     s_arr, eigen_vec = compute_hessian_eig_decomp(x, W_minibatch, f_torch, H_rank + 1)
            # u_vec = eigen_vec.T
        H = self.problem_instance.f_hess(x)
        u_vec, s_arr, _ = np.linalg.svd(H)
        U = u_vec[:, 0:H_rank].dot(np.diag(np.sqrt(s_arr[0:H_rank])))
        diag_H = np.maximum(0, np.diag(H - U.dot(U.T)))
        return U, diag_H

    def low_rank_new_samp_update(self, x, H_rank):
        W_minibatch = None
        if H_rank == self.problem_instance.n:
            print("ERROR: Rank must be less than n for New Samp.")
        if self.problem_instance.n > 10000:
            print("Fast eigen-decomposition for n >= 10000 is commented out.")
            # if W_minibatch is None:
            #     s_arr, eigen_vec = compute_hessian_eig_decomp(x, W, f_torch, H_rank + 1)
            # else:
            #     s_arr, eigen_vec = compute_hessian_eig_decomp(x, W_minibatch, f_torch, H_rank + 1)
            # u_vec = eigen_vec.T
        H = self.problem_instance.f_hess(x)
        u_vec, s_arr, _ = np.linalg.svd(H)
        sigma_NewSamp = np.sqrt(s_arr[0:H_rank] - s_arr[H_rank])
        U = u_vec[:, 0:H_rank].dot(np.diag(sigma_NewSamp))
        diag_H = np.ones(self.problem_instance.n) * s_arr[H_rank]
        return U, diag_H

    def outer_prod_update(self, x, W, num_samples_Jacobian, N, f_Jacobian_value, H_rank):
        if num_samples_Jacobian < N:
            idx = np.random.choice(N, num_samples_Jacobian)
            jacob = f_Jacobian_value(x, W[:, idx])
        else:
            jacob = f_Jacobian_value(x, W)
        u_vec, s_arr, _ = np.linalg.svd(jacob / np.sqrt(num_samples_Jacobian))
        s_arr = np.sqrt(s_arr)
        U = u_vec[:, 0:H_rank].dot(np.diag(s_arr))
        return U

    def Hutchison_update(self, x):
        est_tr = self.problem_instance.f_hess_tr_Hutchinson(x)
        diag_H = est_tr / self.problem_instance.n * np.ones(self.problem_instance.n)
        return diag_H