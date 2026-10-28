from pathlib import Path
import numpy as np, json
from ad_basic import bern_chernoff, xi_QC, output_pair
g = 0.5
print("Exponent table (gamma=0.5)")
for D in [0.1, 0.03, 0.01]:
    h = g-D
    f = np.sqrt(g*h)+np.sqrt((1-g)*(1-h)); mu = np.sqrt(g*h)/(1-np.sqrt((1-g)*(1-h)))
    b = bern_chernoff(g, h)
    A, B = output_pair(g, h, np.eye(2)/2); ch = xi_QC(A, B)[0]
    print("D=%.2f  xi_prod=%.4e (D^2/(8g(1-g))=%.4e)  Choi=%.4e  -ln f=%.4e  -ln f^2=%.4e  -2ln mu=%.4e (D^2/(4g^2(1-g))=%.4e)  ratio UB/LB: conj %.3f proved %.3f" %
          (D, b, D**2/(8*g*(1-g)), ch, -np.log(f), -2*np.log(f), -2*np.log(mu), D**2/(4*g*g*(1-g)), -2*np.log(f)/b, -2*np.log(mu)/b))
print("\nFinite-n table")
res = json.load(open(Path(__file__).resolve().parent / 'sdp_results_0.1_0.03_0.01_0.3.json'))
for r in res:
    D, n = r['Delta'], r['n']; h = g-D
    f = np.sqrt(g*h)+np.sqrt((1-g)*(1-h))
    fl = 0.5*(1-np.sqrt(1-f**(2*n)))
    print("D=%.2f n=%d adapt=%.4f par=%.4f |1>=%.4f  pureLB=%.4f  fidLB=%.4f" % (D, n, r['adapt'], r['parallel'], r['prod1'], r['purelb'], fl))
