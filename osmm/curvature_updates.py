import numpy as np


class CurvatureUpdate:
    def __init__(self, f_torch, hessian_rank, n, n0, n1):
        self.f_torch = f_torch
        self.hessian_rank = hessian_rank
        self.n = n
        self.n0 = n0
        self.n1 = n1

    def low_rank_quasi_Newton_update(self, G_k, xcur, xprev, grad_cur, grad_prev, tol_1=1e-10, tol_2=1e-2, ep=1e-15):
        s = xcur - xprev
        y = grad_cur - grad_prev
        y_abs_too_small_idxes = np.where(np.abs(y) <= ep)[0]
        y[y_abs_too_small_idxes] = 0
        yTs = y.T.dot(s)
        r1 = self.hessian_rank
        w = G_k.T.dot(s)
        if yTs < max(tol_1, np.linalg.norm(y) * np.linalg.norm(s) * tol_2):
            if np.linalg.norm(s) < tol_2 * np.linalg.norm(xprev) or np.linalg.norm(w) < tol_1:
                # print("U_k not updated", np.linalg.norm(s), tol_2 * np.linalg.norm(xprev), np.linalg.norm(v))
                return G_k
            r1 = -1
        while r1 > 0 and yTs - w[0:r1].T.dot(w[0:r1]) < np.linalg.norm(y - G_k[:, 0:r1].dot(w[0:r1])) \
                * np.linalg.norm(s) * tol_2:
            r1 -= 1
        if r1 >= 0:
            w1 = w[0:r1]
            G1 = G_k[:, 0:r1]
            alpha_k = np.sqrt(yTs - w1.T.dot(w1))
            P = np.fliplr(np.eye(r1 + 1))
            tmp = np.eye(r1 + 1)
            tmp[0, 0] = 1.0 / alpha_k
            tmp[1:r1 + 1, 0] = - w1 / alpha_k
            _, R1_tilde = np.linalg.qr(np.transpose(P.dot(tmp)))
            R1 = P.dot(R1_tilde.T.dot(P))
            new_G1 = np.concatenate([y.reshape((self.n, 1)), G1], axis=1).dot(R1)  # n by r1+1
        else:
            new_G1 = None
        if r1 == self.hessian_rank:
            G_k_plus_one = new_G1[:, 0:self.hessian_rank]
        else:
            r2 = self.hessian_rank - max(0, r1)
            w2 = w[self.hessian_rank - r2:self.hessian_rank]
            G2 = G_k[:, self.hessian_rank - r2:self.hessian_rank]
            basis, _, _ = np.linalg.svd(w2.reshape((r2, 1)))
            Q2 = np.array(basis)
            Q2[:, r2 - 1] = basis[:, 0]
            Q2[:, 0:r2 - 1] = basis[:, 1::]
            G2Q2 = G2.dot(Q2)
            new_G2 = G2Q2[:, 0:r2 - 1]  # n by r2-1
            if r1 >= 0:
                G_k_plus_one = np.concatenate([new_G1, new_G2], axis=1)
            else:
                G_k_plus_one = G_k
                G_k_plus_one[:, 0:self.hessian_rank - 1] = new_G2
                G_k_plus_one[:, self.hessian_rank - 1] = np.zeros(self.n)
        return G_k_plus_one

    def low_rank_diag_update(self, x):
        if self.n > 10000:
            print("Fast eigen-decomposition for n >= 10000 not yet supported.")
        H = self.f_torch.f_hess_value(x)
        print(H.shape)
        u_vec, s_arr, _ = np.linalg.svd(H)
        G_k_plus_one = u_vec[:, 0:self.hessian_rank].dot(np.diag(np.sqrt(s_arr[0:self.hessian_rank])))
        if self.hessian_rank == self.n:
            H_diag_k_plus_one = np.zeros(self.n)
        else:
            H_diag_k_plus_one = np.maximum(0, np.diag(H - G_k_plus_one.dot(G_k_plus_one.T)))
        return G_k_plus_one, H_diag_k_plus_one
    #
    # def low_rank_new_samp_update(self, x, H_rank):
    #     if H_rank == self.osmm_problem.n:
    #         print("ERROR: Rank must be less than n for New Samp.")
    #     if self.osmm_problem.n > 10000:
    #         print("Fast eigen-decomposition for n >= 10000 not supported.")
    #     H = self.osmm_problem.f_hess(x)
    #     u_vec, s_arr, _ = np.linalg.svd(H)
    #     sigma_NewSamp = np.sqrt(s_arr[0:H_rank] - s_arr[H_rank])
    #     G_k_plus_one = u_vec[:, 0:H_rank].dot(np.diag(sigma_NewSamp))
    #     diag_H = np.ones(self.osmm_problem.n) * s_arr[H_rank]
    #     return G_k_plus_one, diag_H
    #
    # def BFGS_update(self, G_k, xcur, xprev, grad_cur, grad_prev, tol=1e-10):  # full-rank
    #     s = xcur - xprev
    #     y = grad_cur - grad_prev
    #     yTs = y.T.dot(s)
    #     H_k = G_k.dot(G_k.T)
    #     if yTs > tol:
    #         Hs = H_k.dot(s)
    #         H_k_plus_ones = H_k + np.outer(y, y) / yTs - np.outer(Hs, Hs) / s.T.dot(Hs)
    #         u_vec, s_arr, _ = np.linalg.svd(H_k_plus_ones)
    #         s_arr = np.sqrt(np.maximum(0, s_arr))
    #         G_k_plus_one = u_vec.dot(np.diag(s_arr))
    #     else:
    #         G_k_plus_one = G_k
    #     return G_k_plus_one
    #
    #
    # def trace_hess_update(self, x):
    #     H = self.osmm_problem.f_hess(x)
    #     diag_H = np.trace(H) / self.osmm_problem.n * np.ones(self.osmm_problem.n)
    #     return diag_H
    #
    # def diag_hess_update(self, x):
    #     H = self.osmm_problem.f_hess(x)
    #     diag_H = np.diag(H)
    #     return diag_H
    #
    # def Hutchison_update(self, x):
    #     est_tr = self.osmm_problem.f_hess_tr_Hutchinson(x)
    #     diag_H = est_tr / self.osmm_problem.n * np.ones(self.osmm_problem.n)
    #     return diag_H