import numpy as np, cvxpy as cp, itertools, sys
rng=np.random.default_rng(1)

def schur_norm(M):
    n=M.shape[0]
    Y1=cp.Variable((n,n),symmetric=True); Y2=cp.Variable((n,n),symmetric=True); t=cp.Variable()
    cons=[cp.bmat([[Y1,M],[M.T,Y2]])>>0, cp.diag(Y1)<=t, cp.diag(Y2)<=t]
    p=cp.Problem(cp.Minimize(t),cons); p.solve(solver=cp.CLARABEL)
    return t.value

def dd(h,hp,pts):
    n=len(pts); M=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if abs(pts[i]-pts[j])<1e-12: M[i,j]=hp(pts[i])
            else: M[i,j]=(h(pts[i])-h(pts[j]))/(pts[i]-pts[j])
    return M

fs={
 'Hellinger': (lambda t:1-2*np.sqrt(t*(1-t)), lambda t:-(1-2*t)/np.sqrt(t*(1-t))),
 'LeCam': (lambda t:(2*t-1)**2, lambda t:4*(2*t-1)),
 'JS': (lambda t:np.log(2)+t*np.log(t)+(1-t)*np.log(1-t), lambda t:np.log(t/(1-t))),
 'Jeffreys': (lambda t:(2*t-1)*np.log(t/(1-t)), lambda t:2*np.log(t/(1-t))+(2*t-1)/(t*(1-t))),
 'Neyman': (lambda t:(2*t-1)**2/t, lambda t:4-1/t**2),
 'Pearson': (lambda t:(2*t-1)**2/(1-t), lambda t:1/(1-t)**2-4),
 'symchi2': (lambda t:1/(t*(1-t))-4, lambda t:(2*t-1)/(t**2*(1-t)**2)),
 'BS': (lambda t:t*np.log(t/(1-t)), lambda t:np.log(t/(1-t))+1/(1-t)),
 'Renyi1.5': (lambda t:t**1.5*(1-t)**-0.5, lambda t:t**0.5*(1-t)**-1.5*(1.5-t)),
}
table={
 'Hellinger': lambda e:(1-2*e)/np.sqrt(e*(1-e)),
 'LeCam': lambda e:4*(1-2*e),
 'JS': lambda e:np.log((1-e)/e),
 'Jeffreys': lambda e:2*np.log((1-e)/e)+(1-2*e)/(e*(1-e)),
 'Neyman': lambda e:e**-2-4,
 'Pearson': lambda e:e**-2-4,
 'symchi2': lambda e:e**-2-(1-e)**-2,
 'BS': lambda e:np.log((1-e)/e)+1/e,
 'Renyi1.5': lambda e:(0.5+e)*(1-e)**0.5/e**1.5,
}
for eps in [0.2,0.05]:
    print("eps",eps)
    for name,(h,hp) in fs.items():
        a,b=eps,1-eps
        grid=np.linspace(a,b,4001)
        maxhp=np.max(np.abs(hp(grid)))
        end=schur_norm(dd(h,hp,[a,b]))
        best=end
        for k in range(30):
            m=rng.integers(3,7)
            pts=np.sort(np.concatenate([[a,b],rng.uniform(a,b,m-2)]))
            best=max(best,schur_norm(dd(h,hp,list(pts))))
        print(f"{name:10s} table={table[name](eps):10.5f} max|h'|={maxhp:10.5f} endpoint-Schur={end:10.5f} best-over-random-sets={best:10.5f}")
