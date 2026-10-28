import numpy as np, mpmath as mp
from qlib import *
mp.mp.dps=80
def JAD(g):
    g=mp.mpf(g)
    v0=[1,0,0,mp.sqrt(1-g)]; v1=[0,mp.sqrt(g),0,0]  # vec ordering (y,x): |00>,|01>,|10>,|11>
    return mpouter(v0)+mpouter(v1)
for g in ['0.5','0.9','1']:
    for d in ['1e-3','1e-6']:
        gp=mp.mpf(g)-mp.mpf(d)
        fs=[fid_mp(JAD(g),JAD(gp),2,2,e,dps=80) for e in ['1e-20','1e-30','1e-40']]
        print(f"g={g} g'={mp.nstr(gp,8)}: F_eps(1e-20,1e-30,1e-40)={[mp.nstr(x,4) for x in fs]}")
    ff=fid_mp(JAD(g),JAD(g),2,2,'1e-40',dps=80)
    # double precision naive (to show failure)
    Jn=mp2np(JAD(g)); Jnp=mp2np(JAD(mp.mpf(g)-mp.mpf('1e-6')))
    try: nv=fid_np(Jn+1e-12*np.eye(4),Jnp+1e-12*np.eye(4),2,2)
    except Exception as e: nv=str(e)
    print(f"  F(AD_g,AD_g)={mp.nstr(ff,15)} ; double-precision eps=1e-12 gives {nv}")
# analytic intersection check: kernel of [vecK0(g), e01, -vecK0(g'), -e01]
for g,gp in [(0.5,0.499),(1,0.999)]:
    A=np.array([[1,0,1,0],[0,1,0,1],[0,0,0,0],[np.sqrt(1-g),0,np.sqrt(1-gp),0]])
    M=np.column_stack([A[:,0],A[:,1],-A[:,2],-A[:,3]])
    print('null dim of [range A | -range B]:',4-np.linalg.matrix_rank(M))
