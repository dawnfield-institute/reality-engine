# RED–PAC–HP triad experiment
# Combines RED spectral entropy metrics, PAC re-actualization dynamics, and HP information recoverability
# to test conservation laws in the Dawn Field framework.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eigvalsh
from math import log2
from scipy.stats import spearmanr, pearsonr

# ---- Helper Functions ----
def kron_all(ops):
    M = np.array([[1]], dtype=complex)
    for o in ops:
        M = np.kron(M, o)
    return M

def projector_state(vec):
    v = vec.reshape(-1,1)
    return v @ v.conj().T

def random_single():
    A = np.random.normal(size=(2,2)) + 1j*np.random.normal(size=(2,2))
    Q, R = np.linalg.qr(A)
    d = np.diag(R); L = np.diag(d/np.abs(d))
    return Q @ L

def random_two_qubit_unitary():
    A = np.random.normal(size=(4,4)) + 1j*np.random.normal(size=(4,4))
    Q, R = np.linalg.qr(A)
    d = np.diag(R); L = np.diag(d/np.abs(d))
    return Q @ L

# Embedding utilities
def embed_local_unitary(U1, q, n_total):
    Uq = np.eye(2**n_total, dtype=complex)
    for b in range(2**n_total):
        bits_b = [(b>>t)&1 for t in reversed(range(n_total))]
        for c in range(2**n_total):
            bits_c = [(c>>t)&1 for t in reversed(range(n_total))]
            ok = True
            for t in range(n_total):
                if t!=q and bits_b[t]!=bits_c[t]:
                    ok=False; break
            if not ok: continue
            Uq[b,c] = U1[bits_b[q], bits_c[q]]
    return Uq

def embed_subsystem_unitary(U_sub, S_qubits, n_total):
    U_full = np.eye(2**n_total, dtype=complex)
    for a in range(2**n_total):
        bits_a = [(a>>b)&1 for b in reversed(range(n_total))]
        for b in range(2**n_total):
            bits_b = [(b>>t)&1 for t in reversed(range(n_total))]
            ok=True
            for q in range(n_total):
                if q not in S_qubits and bits_a[q]!=bits_b[q]:
                    ok=False; break
            if not ok: continue
            idx_a = 0; idx_b = 0
            for q in S_qubits:
                idx_a = (idx_a<<1) | bits_a[q]
                idx_b = (idx_b<<1) | bits_b[q]
            U_full[a,b] = U_sub[idx_a, idx_b]
    return U_full

# Partial trace and entropy
def partial_trace(rho, keep, dims):
    n = len(dims)
    keep = sorted(keep)
    trace_idx = [i for i in range(n) if i not in keep]
    rho_t = rho.reshape(*(dims + dims))
    order = keep + trace_idx + [i+n for i in keep] + [i+n for i in trace_idx]
    rho_p = np.transpose(rho_t, axes=order)
    d_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    d_trace = int(np.prod([dims[i] for i in trace_idx])) if trace_idx else 1
    rho_p = rho_p.reshape(d_keep, d_trace, d_keep, d_trace)
    rho_red = np.einsum('abcb->ac', rho_p)
    return rho_red

def von_neumann_entropy(rho):
    vals = np.clip(np.real(eigvalsh((rho + rho.conj().T)/2)), 0, 1)
    vals = vals/np.sum(vals) if vals.sum()>0 else vals
    vals = vals[vals>1e-14]
    H = -(vals*np.log2(vals)).sum() if len(vals)>0 else 0.0
    return float(H)

def mutual_information(rho, A, B, dims):
    rhoA = partial_trace(rho, A, dims)
    rhoB = partial_trace(rho, B, dims)
    rhoAB = partial_trace(rho, sorted(set(A+B)), dims)
    return von_neumann_entropy(rhoA)+von_neumann_entropy(rhoB)-von_neumann_entropy(rhoAB)

# Generate random HP-like instance
def hp_instance(n_total=8, k_message=2, n_B=4, depth=5, m_R_emit=2):
    dims = [2]*n_total
    psi0 = np.zeros((2**n_total,), dtype=complex); psi0[0]=1.0
    rho = projector_state(psi0)
    targets = list(range(k_message)) + list(range(k_message, k_message+n_B))
    U_total = np.eye(2**n_total, dtype=complex)
    for q in targets:
        U_total = embed_local_unitary(random_single(), q, n_total) @ U_total
    rho = U_total @ rho @ U_total.conj().T
    S_qubits = list(range(0, k_message+n_B))
    U_S = np.eye(2**len(S_qubits), dtype=complex)
    for _ in range(depth):
        U_S = random_two_qubit_unitary() @ U_S
    U_full = embed_subsystem_unitary(U_S, S_qubits, n_total)
    rho = U_full @ rho @ U_full.conj().T
    R_qubits = sorted(np.random.choice(S_qubits, m_R_emit, replace=False))
    M_qubits = list(range(0, k_message))
    I_MR = mutual_information(rho, M_qubits, R_qubits, dims)
    rho_R = partial_trace(rho, R_qubits, dims)
    evals_R = np.sort(np.real(eigvalsh((rho_R + rho_R.conj().T)/2)))[::-1]
    return {"rho":rho,"dims":dims,"M":M_qubits,"R":R_qubits,"I_MR":I_MR,"evals_R":evals_R}

# RED metrics
def red_metrics_from_spectrum(evals):
    p = np.clip(np.real(evals), 1e-16, None)
    p = p / p.sum() if p.sum()>0 else p
    H = -(p*np.log2(p)).sum() if p.sum()>0 else 0.0
    Hn = H / log2(len(p)) if len(p)>1 else 0.0
    pr = 1.0 / float((p**2).sum()) if p.sum()>0 else 1.0
    d = len(p)
    pr_n = (pr - 1.0) / (d - 1.0) if d>1 else 0.0
    red_eff = (1.0 - Hn) * (1.0 - pr_n)
    return {"RED_Hn": Hn, "RED_PRn": pr_n, "RED_eff": red_eff}

# PAC dynamics
def PAC_reactualize(W, steps=140, alpha=0.6, beta=0.05, gamma=0.4, lambda_=0.5, mu=0.02, Kmax=10.0, seed=0):
    np.random.seed(seed)
    n = W.shape[0] if W.ndim==2 else 1
    a = np.clip(np.random.normal(0.06, 0.02, size=n), 0, 1)
    p = np.abs(np.random.normal(0.35, 0.06, size=n))
    deg = W.sum(axis=1) if n>1 else np.array([0.1])
    phi = (deg - deg.min()) / (deg.max() - deg.min() + 1e-9) + 0.1
    A_hist=[]; P_hist=[]; K_hist=[]
    for t in range(steps):
        lap = (W @ a - np.diag(W.sum(axis=1)) @ a) if n>1 else np.array([0.0])
        a_update = alpha*phi*(1-a) - beta*a + gamma*lap
        p_update = phi - lambda_*a - mu*p
        a = np.clip(a + 0.05*a_update, 0, 1)
        p = np.clip(p + 0.05*p_update, 0, None)
        K = p.sum() + a.sum()
        if K > Kmax:
            scale = Kmax / (K + 1e-9)
            a *= scale; p *= scale; K = p.sum() + a.sum()
        A_hist.append(a.sum()); P_hist.append(p.sum()); K_hist.append(K)
    return np.array(A_hist), np.array(P_hist), np.array(K_hist)

# Mutual info matrix
def quantum_mi_matrix_over_R(rho, R, dims):
    m = len(R)
    if m <= 1:
        return np.zeros((m,m))
    W = np.zeros((m, m))
    for a in range(m):
        for b in range(a+1, m):
            IA_B = mutual_information(rho, [R[a]], [R[b]], dims)
            W[a, b] = IA_B; W[b, a] = IA_B
    if W.max() > 0: W = W / W.max()
    np.fill_diagonal(W, 0.0)
    return W

# ---- Triad Experiment ----
def build_triad(depths=(3,5), R_emit=(1,2,3), runs=3, Kmax=9.0):
    rows = []
    for d in depths:
        for rsz in R_emit:
            for s in range(runs):
                inst = hp_instance(n_total=6, k_message=1, n_B=3, depth=d, m_R_emit=rsz)
                rho = inst["rho"]; dims = inst["dims"]; R = inst["R"]
                I_MR = inst["I_MR"]
                rho_R = partial_trace(rho, R, dims)
                evals_R = np.sort(np.real(eigvalsh((rho_R + rho_R.conj().T)/2)))[::-1]
                redm = red_metrics_from_spectrum(evals_R)
                W_R = quantum_mi_matrix_over_R(rho, R, dims)
                A_hist, _, _ = PAC_reactualize(W_R, Kmax=Kmax, seed=300+s)
                PAC_A = float(A_hist[-1])
                C_F = PAC_A / (redm["RED_eff"] + 1e-9)
                rows.append({"depth": d, "R_emit": rsz, "run": s,
                             "I_MR": I_MR, "PAC_A_final": PAC_A,
                             "RED_Hn": redm["RED_Hn"], "RED_PRn": redm["RED_PRn"],
                             "RED_eff": redm["RED_eff"], "C_F": C_F, "R_size": len(R)})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = build_triad()
    print(df.head())
    # Correlations
    c1 = spearmanr(df["RED_eff"], df["I_MR"])
    c2 = spearmanr(df["RED_eff"], df["PAC_A_final"])
    c3 = spearmanr(df["PAC_A_final"], df["I_MR"])
    print("Correlations:")
    print(f"RED_eff vs I_MR: {c1}")
    print(f"RED_eff vs PAC_A_final: {c2}")
    print(f"PAC_A_final vs I_MR: {c3}")


# --- Upgrades: modular scramblers, noise models, Page sweep, and Petz-style proxy ---
# You can run these helpers below the original triad code. They are drop-in compatible.

import itertools

# 1) Scrambler choices: 'haar_layers' (default), 'clifford_like', 'syk_proxy'

def random_clifford_like_two_qubit():
    # Small fixed gate set approximating a Clifford layer
    H = (1/np.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)
    S = np.array([[1,0],[0,1j]], dtype=complex)
    CX = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,0,1],
                   [0,0,1,0]], dtype=complex)
    # Randomly pick a pattern H⊗I, I⊗H, S⊗S, then CX
    choices = [np.kron(H, np.eye(2)), np.kron(np.eye(2), H), np.kron(S, S)]
    U = choices[np.random.randint(0,len(choices))] @ CX
    return U


def syk_proxy_two_qubit():
    # Lightweight SYK-like 2-qubit: random anti-Hermitian scaled exponentiation
    J = np.random.normal(size=(4,4))
    A = J - J.T  # anti-symmetric real -> anti-Hermitian after multiplying by i
    H = 1j * A
    # expm via truncated series (sufficient for small norms)
    I4 = np.eye(4, dtype=complex)
    U = I4 + H + 0.5*(H@H) + (1/6.0)*(H@H@H)
    # QR to re-unitarize
    Q,R = np.linalg.qr(U)
    d = np.diag(R); L = np.diag(d/np.abs(d))
    return Q @ L


def apply_random_layers_mod(n_qubits, depth=6, mode='haar_layers'):
    pairs = [(i,j) for i in range(n_qubits) for j in range(i+1,n_qubits)]
    U = np.eye(2**n_qubits, dtype=complex)
    for _ in range(depth):
        i,j = pairs[np.random.randint(0, len(pairs))]
        if mode=='haar_layers':
            U2 = random_two_qubit_unitary()
        elif mode=='clifford_like':
            U2 = random_clifford_like_two_qubit()
        elif mode=='syk_proxy':
            U2 = syk_proxy_two_qubit()
        else:
            U2 = random_two_qubit_unitary()
        # Embed U2 on (i,j) inside n_qubits space (reuse approach from apply_random_layers)
        U_full = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
        for a in range(2**n_qubits):
            ai = (a>>(n_qubits-1-i))&1; aj = (a>>(n_qubits-1-j))&1
            for b in range(2**n_qubits):
                bi = (b>>(n_qubits-1-i))&1; bj = (b>>(n_qubits-1-j))&1
                ok=True
                for q in range(n_qubits):
                    if q in (i,j): continue
                    if ((a>>(n_qubits-1-q))&1) != ((b>>(n_qubits-1-q))&1):
                        ok=False; break
                if not ok: continue
                idx_a = ai*2 + aj; idx_b = bi*2 + bj
                U_full[a,b] = U2[idx_a, idx_b]
        U = U_full @ U
    return U

# 2) Noisy emission: depolarizing channel on R before MI/RED

def depolarize_rho_on_subset(rho, qubits, dims, p=0.0):
    if p <= 0.0 or len(qubits)==0:
        return rho
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    K = [np.sqrt(1-3*p/4)*I2, np.sqrt(p/4)*X, np.sqrt(p/4)*Y, np.sqrt(p/4)*Z]
    out = rho
    for q in qubits:
        lifted = []
        for k in K:
            ops = [I2]*len(dims)
            ops[q] = k
            lifted.append(kron_all(ops))
        new_out = np.zeros_like(out, dtype=complex)
        for L in lifted:
            new_out += L @ out @ L.conj().T
        out = new_out
    return out

# 3) Petz-style proxy decoder score (very lightweight)
# We approximate a recovery score by comparing I(M:R) vs I(M:RB) where B is a minimal extra qubit from the black hole region.
# Higher delta suggests easier recovery.

def petz_proxy_score(rho, dims, M, R, B_extra):
    I_MR = mutual_information(rho, M, R, dims)
    R_ext = sorted(list(set(R + B_extra)))
    I_MRext = mutual_information(rho, M, R_ext, dims)
    return I_MRext - I_MR

# 4) Page-like sweep runner with options

def run_page_sweep(n_total=8, k_message=2, n_B=4, depths=(3,5,7), R_emit=(1,2,3,4), runs=5,
                   scrambler='haar_layers', noise_p=0.0, Kmax=10.0, use_petz=False):
    rows = []
    for d in depths:
        for rsz in R_emit:
            for s in range(runs):
                # Build HP instance with modular scrambler
                dims = [2]*n_total
                psi0 = np.zeros((2**n_total,), dtype=complex); psi0[0]=1.0
                rho = projector_state(psi0)
                targets = list(range(k_message)) + list(range(k_message, k_message+n_B))
                U_total = np.eye(2**n_total, dtype=complex)
                for q in targets:
                    U_total = embed_local_unitary(random_single(), q, n_total) @ U_total
                rho = U_total @ rho @ U_total.conj().T
                S_qubits = list(range(0, k_message+n_B))
                U_S = apply_random_layers_mod(n_qubits=len(S_qubits), depth=d, mode=scrambler)
                U_full = embed_subsystem_unitary(U_S, S_qubits, n_total)
                rho = U_full @ rho @ U_full.conj().T
                # Emit radiation
                R_qubits = sorted(np.random.choice(S_qubits, rsz, replace=False))
                M_qubits = list(range(0, k_message))
                # Optional noise on R
                rho_eff = depolarize_rho_on_subset(rho, R_qubits, dims, p=noise_p)
                # HP
                I_MR = mutual_information(rho_eff, M_qubits, R_qubits, dims)
                # Petz proxy
                petz = None
                if use_petz:
                    # take one extra qubit from the remaining black hole region if possible
                    BH_remaining = [q for q in S_qubits if q not in R_qubits and q not in M_qubits]
                    B_extra = BH_remaining[:1]
                    petz = petz_proxy_score(rho_eff, dims, M_qubits, R_qubits, B_extra) if B_extra else 0.0
                # RED
                rho_R = partial_trace(rho_eff, R_qubits, dims)
                evals_R = np.sort(np.real(eigvalsh((rho_R + rho_R.conj().T)/2)))[::-1]
                redm = red_metrics_from_spectrum(evals_R)
                # PAC
                W_R = quantum_mi_matrix_over_R(rho_eff, R_qubits, dims)
                A_hist, _, _ = PAC_reactualize(W_R, Kmax=Kmax, seed=900+s)
                PAC_A = float(A_hist[-1])
                rows.append({
                    'depth': d, 'R_emit': rsz, 'run': s, 'scrambler': scrambler,
                    'noise_p': noise_p, 'Kmax': Kmax,
                    'I_MR': I_MR, 'PAC_A_final': PAC_A,
                    'RED_eff': redm['RED_eff'], 'RED_Hn': redm['RED_Hn'], 'RED_PRn': redm['RED_PRn'],
                    'petz_proxy': petz,
                })
    return pd.DataFrame(rows)

# 5) Convenience plotters

def plot_page(df, title_prefix=""):
    g = df.groupby('R_emit').agg(I_med=('I_MR','median'), PAC_med=('PAC_A_final','median')).reset_index()
    plt.figure(figsize=(6,4)); plt.plot(g['R_emit'], g['I_med'], marker='o');
    plt.xlabel('|R|'); plt.ylabel('Median I(M:R)'); plt.title(f'{title_prefix} Page-like curve: I(M:R)'); plt.tight_layout(); plt.show()
    plt.figure(figsize=(6,4)); plt.plot(g['R_emit'], g['PAC_med'], marker='o');
    plt.xlabel('|R|'); plt.ylabel('Median PAC Σa_i'); plt.title(f'{title_prefix} PAC vs |R|'); plt.tight_layout(); plt.show()


def correlation_table(df):
    def corr_col(x,y):
        rs, ps = spearmanr(df[x], df[y]); rp, pp = pearsonr(df[x], df[y]); return rs, ps, rp, pp
    return {
        'RED↔PAC': corr_col('RED_eff','PAC_A_final'),
        'RED↔HP' : corr_col('RED_eff','I_MR'),
        'PAC↔HP' : corr_col('PAC_A_final','I_MR'),
        'Petz↔PAC': corr_col('petz_proxy','PAC_A_final') if 'petz_proxy' in df else None,
    }

# Example usage (run locally):
 df = run_page_sweep(n_total=8, k_message=2, n_B=4, depths=(3,5,7,9), R_emit=(1,2,3,4), runs=6,
#                     scrambler='syk_proxy', noise_p=0.02, Kmax=10.0, use_petz=True)
 print(correlation_table(df))
 plot_page(df, title_prefix='SYK proxy, p=0.02')
