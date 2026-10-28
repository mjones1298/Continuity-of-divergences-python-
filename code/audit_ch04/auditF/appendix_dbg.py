import numpy as np, sys
sys.path.insert(0,'.')
from lib import *
from _helpers3 import *
rng = np.random.default_rng(15)
conns = make_conns()
worst=(0,None)
for trial in range(400):
    dY = rng.integers(1, 4); dX = rng.integers(1, 4)
    KP = random_kraus(dY, dX, rng.integers(1, dY*dX+1), rng); KQ = random_kraus(dY, dX, rng.integers(1, dY*dX+1), rng)
    JP = sum(np.outer(vec(K), vec(K).conj()) for K in KP); JQ = sum(np.outer(vec(K), vec(K).conj()) for K in KQ); JO = JP + JQ
    KO = kraus_from_choi(JO, dY, dX)
    w, V = np.linalg.eigh(JO); keep = w > 1e-10
    Wi = V[:, keep]/np.sqrt(w[keep]); T = herm(Wi.conj().T@JP@Wi); wt = np.clip(np.linalg.eigvalsh(T), 0, 1)
    for name in ['geo','harm','alpha0.3','alpha0.8','log','mix','arith','skew']:
        cn = conns[name]
        M = ka_frame_conn(JP, JQ, cn, tol=1e-12); KM = kraus_from_choi(M, dY, dX)
        gmax = cn.phi(wt).max()
        for nm,nf in (('1',p1),('inf',pinf),('2',p2)):
            rhs = gmax*nf(KO)
            if rhs>0:
                r = nf(KM)/rhs
                if r>worst[0]: worst=(r,(trial,name,nm,dY,dX,len(KP),len(KQ),nf(KM),rhs,gmax,nf(KO),wt, np.linalg.norm(M-M.conj().T)))
print(worst)
