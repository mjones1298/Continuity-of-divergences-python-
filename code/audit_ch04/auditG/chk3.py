import numpy as np, cvxpy as cp
exec(open('chk2.py').read().split("fs={")[0])
h=lambda t:(2*t-1)**2/(1-t); hp=lambda t:4-1/(1-t)**2
eps=0.2; a,b=eps,1-eps
rng=np.random.default_rng(1)
# replay random sets to find the offender
worst=0;wp=None
for k in range(300):
    m=rng.integers(3,7)
    pts=np.sort(np.concatenate([[a,b],rng.uniform(a,b,m-2)]))
    M=dd(h,hp,list(pts))
    for solver in [cp.CLARABEL]:
        v=schur_norm(M)
    if v>worst: worst=v;wp=pts
print(worst,wp)
M=dd(h,hp,list(wp))
for sname in ['CLARABEL','SCS']:
    n=M.shape[0]
    Y1=cp.Variable((n,n),symmetric=True); Y2=cp.Variable((n,n),symmetric=True); t=cp.Variable()
    p=cp.Problem(cp.Minimize(t),[cp.bmat([[Y1,M],[M.T,Y2]])>>0, cp.diag(Y1)<=t, cp.diag(Y2)<=t])
    try:
        p.solve(solver=getattr(cp,sname),eps=1e-9) if sname=='SCS' else p.solve(solver=cp.CLARABEL,tol_gap_abs=1e-12,tol_gap_rel=1e-12,tol_feas=1e-12)
        print(sname,t.value,p.status)
    except Exception as e: print(sname,'err',e)
# lower bound check: ||M o Z|| for explicit random contractions Z? and the Haagerup factorization check
print(np.linalg.eigvalsh(M))
print(np.max(np.abs(M)))
