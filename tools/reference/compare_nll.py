import numpy as np, sys
def rd(p):
    W={};c=None
    for l in open(p):
        if l.startswith('w '): c=int(l.split()[1])
        elif l.startswith('lp ') and c is not None: W[c]=np.array(l.split()[1:],float)
    return W
a=rd(sys.argv[1]); b=rd(sys.argv[2])
for w in sorted(a):
    if w in b: d=a[w]-b[w]; print(w,'ppl_a %.3f ppl_b %.3f mean|dlogp| %.4f max %.3f'%(np.exp(-a[w].mean()),np.exp(-b[w].mean()),abs(d).mean(),abs(d).max()))
