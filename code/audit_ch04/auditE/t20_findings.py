# Numerical illustrations for the MINOR findings of the auditE report (Sections 10-11).
import numpy as np
from lib import *

def vec(*c): return np.array(c,dtype=complex).reshape(-1,1)
def proj(v): v=v/np.linalg.norm(v); return v@v.conj().T

print("=== F1: Q_R = 0 (orthogonal supports), d_ac convention")
dY,dX=2,1
e1=vec(1,0); e2=vec(0,1)
J=[proj(e1),proj(e1),proj(e2),proj(e2)]          # Phi1=Psi1=e1e1*, Phi2=Psi2=e2e2*
S=Setup(*J,dY,dX)
q=sec11(S)
F1=Fhat(J[0],J[1],dY,dX);F2=Fhat(J[2],J[3],dY,dX)
nO=S.nOmb
print(" dims Q1,QR,Q2 =",[int(round(np.trace(q[k]).real)) for k in ('Q1','QR','Q2')],
      " F1,F2 =",F1,F2," ||Omegabar*||=",nO," m_s=",q['ms']," tau_ac,kap1,kap2=",q['tau_ac'],q['kap1'],q['kap2'],"h_ac=",q['hac'])
base=nO*np.sqrt(q['tau_ac'])+0.5*q['ms']+S.nOm[0]*np.sqrt(q['kap1'])+S.nOm[1]*np.sqrt(q['kap2'])
for name,tanh in [("d_ac:=0 (tanh=0)",0.0),("d_T(0,0)=log0=-inf (tanh=-1)",-1.0)]:
    mm=min(np.sqrt(2)*nO*tanh, np.sqrt(nO)*q['hac'])
    print("  R_{g#} with",name,"=",base+mm,"  LHS =",abs(F1-F2))

print("=== F2: equal supports, h_ac -> 0 although d_T fixed (hypothesis of Cor 11.6 is min{d_T,h_ac}->0)")
s=0.25
for eps in [1,1e-1,1e-2,1e-3,1e-4]:
    P1=0.5*np.diag([1,eps]).astype(complex); P2=0.5*np.diag([1,eps*s]).astype(complex)
    S=Setup(P1,P1,P2,P2,2,1)
    q=sec11(S)
    dT=2*np.arctanh(q['tanh_ac'])
    print(f" eps={eps:g}: d_T={dT:.4f} (log(1/s)={np.log(1/s):.4f})  h_ac={q['hac']:.4e}  tau={q['tau_ac']:.1e}  |dF|={abs(Fhat(P1,P1,2,1)-Fhat(P2,P2,2,1)):.4e}")

print("=== F3: d_ac/h_ac cannot be dropped (all other quantities vanish)")
for s in [0.5,0.2]:
    P1=0.5*np.eye(2,dtype=complex); P2=0.5*np.diag([1,s]).astype(complex)
    S=Setup(P1,P1,P2,P2,2,1); q=sec11(S)
    print(f" s={s}: tau_ac={q['tau_ac']:.1e} kap={q['kap1']:.1e},{q['kap2']:.1e} m_s={q['ms']:.1e}  d_ac={2*np.arctanh(q['tanh_ac']):.4f} h_ac={q['hac']:.4f}  |dF|={abs(Fhat(P1,P1,2,1)-Fhat(P2,P2,2,1)):.4f} (=(1-s)/2={(1-s)/2:.4f})")

print("=== F4: unnumbered example after Rem 11.5: y2=-y1 is 'y1!=y2' but the example collapses")
y1=vec(1,0); y2=-y1
print(" F(Phi2,Psi2) with y2=-y1:",Fhat(proj(y1),proj(y2),2,1)," (the text needs F=0)")
th=0.05; y2=vec(np.cos(th),np.sin(th))
J=[proj(y1),proj(y1),proj(y1),proj(y2)]
S=Setup(*J,2,1);q=sec11(S)
print(" y2 at angle 0.05: F1,F2=",Fhat(J[0],J[1],2,1),Fhat(J[2],J[3],2,1)," m_s=",q['ms']," tau_ac=",q['tau_ac'], " C1 eigs",np.round(np.linalg.eigvalsh(S.C[0]),4))

print("=== F5: scalar (commuting) case: sup tau = 2*delta exactly")
rng=np.random.default_rng(0)
for d in [0.4,0.2,0.05]:
    best=0
    for _ in range(400000):
        g=rng.uniform(-d,d)*1.0; c1=0.5+g;c2=0.5-g
        t1,t2=rng.random(2)
        if rng.random()<0.5: t1,t2=rng.choice([0,1],2)
        X=c1*t1-c2*t2;Y=c1*(1-t1)-c2*(1-t2)
        dl=max(abs(X),abs(Y))
        if dl<=d and dl>0: best=max(best,abs(t1-t2)/dl*d)
    print(f" delta<={d}: max tau at delta=d ~ {best:.4f}  (2d={2*d})")

print("=== F6: lower bound sqrt(2) for the constant in Cor 10.7 (scalar example)")
for dlt in [1e-2,1e-4]:
    F1=np.sqrt(dlt*(1-dlt));F2=0; nOb=1.0; dl=dlt/2
    print(f" T1={dlt:g},T2=0: |dF|/(||Omegabar*|| delta^(1/2)) = {abs(F1-F2)/(nOb*np.sqrt(dl)):.5f}  (sqrt2={np.sqrt(2):.5f})")
