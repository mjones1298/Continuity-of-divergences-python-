# Refit opposite-side factor-2 exponents from A_cert README table values, over two p-ranges
import numpy as np
p=np.array([1e-2,3e-3,1e-3,3e-4,1e-4])
rows={"(a) ball F":[4.66e8,4.86e9,4.27e10,4.71e11,4.23e12],
"(a') Hoeff box F":[1.70e7,1.58e8,1.32e9,1.41e10,1.22e11],
"(a'') CP box F":[9.20e5,3.49e6,9.36e6,3.03e7,7.97e7],
"(b) Pauli F":[21800,67500,2.38e5,6.25e5,2.12e6],
"(a) ball BS":[6.04e8,6.32e9,5.53e10,6.13e11,5.50e12],
"(a') box BS":[2.11e7,2.01e8,1.68e9,1.80e10,1.58e11],
"(a'') CP BS":[9.75e5,3.65e6,1.00e7,3.24e7,8.87e7],
"(b) Pauli BS":[28300,87400,3.22e5,8.02e5,2.67e6]}
for k,v in rows.items():
    v=np.array(v)
    a_all=-np.polyfit(np.log(p),np.log(v),1)[0]
    a_4=-np.polyfit(np.log(p[:4]),np.log(v[:4]),1)[0]
    print(f"{k:22s} all p: {a_all:.3f}   p in [3e-4,1e-2]: {a_4:.3f}")
