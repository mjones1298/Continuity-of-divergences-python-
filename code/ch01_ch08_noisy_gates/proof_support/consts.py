import mpmath as mp, numpy as np
mp.mp.dps=30
f=lambda u: u*mp.e**(-u)*mp.log((1+3*mp.e**(-u))/(1-mp.e**(-u)))
u=mp.findroot(lambda u: mp.diff(f,u),0.47); print('R7: u*=',u,' f*=',f(u),' f*/4=',f(u)/4)
# finite p: sup_k (1/k)(1-p_k) sin^2(k d/2) log((4-3p_k)/p_k) * p/d^2 at d->0
for p in [1e-2,1e-3,1e-4]:
    best=0
    for k in range(1,int(3/p)):
        pk=1-(1-p)**k; val=(1/k)*(1-pk)*(k**2/4)*np.log((4-3*pk)/pk)*p
        best=max(best,val)
    print(f'p={p}: sup_k per-use D * p/delta^2 (delta->0) = {best:.6f}')
# two-point constant c0: N >= (ln2/3)/KL, worst over eta<=0.1, p<=0.04 of N*eta^2*lam~
worst=1e9
for p in np.geomspace(1e-6,0.04,40):
    for eta in np.linspace(0.005,0.1,40):
        lt=p/4; s=lt*(((1+eta)/(1-eta))**2-1)
        q=np.array([1-3*p/4,p/4,p/4,p/4]); q2=q.copy(); q2[1]+=s; q2[0]-=s
        KL=np.sum(q*np.log(q/q2)); worst=min(worst,np.log(2)/3/KL*eta**2*lt)
print('two-point: min over grid of N_min*eta^2*lam~ =',worst)
# coherent two-point constant
print('coherent two-point: ln(1/sqrt(8/9)) =',np.log(1/np.sqrt(8/9)))
# unitary-Phi absolute sup limit
print('2/(3sqrt3)=',2/(3*np.sqrt(3)),' 1/(6sqrt3)=',1/(6*np.sqrt(3)))
# window constant: g^{-1}(eta)
for eta in [0.1,0.01]:
    xs=1/(1-eta)**2-1; print(f'eta={eta}: x*={xs:.6f}, delta*/(p/4)^(w/2) -> 4 sqrt(x*) = {4*np.sqrt(xs):.6f}')
