import mpmath as mp
mp.mp.dps=30
pi=mp.pi
def rep(a,b,c,dens,x,atoms=(),):
    # f(x)=a+b(x-1)+c(x-1)^2 + int (x-1)^2/(x+s) dens(s) ds + sum atoms w*(x-1)^2/(x+s0)
    val=a+b*(x-1)+c*(x-1)**2
    I=mp.quad(lambda s:(x-1)**2/(x+s)*dens(s),[0,1,10,mp.inf]) if dens else 0
    for (s0,w) in atoms: val+=w*(x-1)**2/(x+s0)
    return val+I
rows={}
# name: (f, a,b,c, dens, atoms)
al=mp.mpf('1.37'); be=mp.mpf('-0.4')
rows['BS']=(lambda x:x*mp.log(x),0,1,0,lambda s:s/(1+s)**2,())
rows['revBS']=(lambda x:-mp.log(x),0,-1,0,lambda s:1/(1+s)**2,())
rows['Jeffreys']=(lambda x:(x-1)*mp.log(x),0,0,0,lambda s:1/(1+s),())
rows['Pearson']=(lambda x:(x-1)**2,0,0,1,None,())
rows['Neyman']=(lambda x:(x-1)**2/x,0,0,0,None,((0,1),))
rows['symchi2']=(lambda x:(x-1)**2*(1+1/x),0,0,1,None,((0,1),))
rows['LeCam']=(lambda x:(x-1)**2/(x+1),0,0,0,None,((1,1),))
rows['Hellinger']=(lambda x:(mp.sqrt(x)-1)**2,0,0,0,lambda s:2/pi*mp.sqrt(s)/(1+s)**2,())
rows['JS']=(lambda x:x*mp.log(x)-(1+x)*mp.log((1+x)/2),0,0,0,lambda s:min(s,1)/(1+s)**2,())
rows['Renyi a=1.37']=(lambda x:x**al,1,al,0,lambda s:mp.sin(pi*(al-1))/pi*s**al/(1+s)**2,())
rows['Renyi a=-0.4']=(lambda x:x**be,1,be,0,lambda s:-mp.sin(pi*be)/pi*s**be/(1+s)**2,())
for name,(f,a,b,c,dens,atoms) in rows.items():
    errs=[]
    for x in [mp.mpf('0.3'),mp.mpf('2.5'),mp.mpf('7')]:
        errs.append(mp.nstr(f(x)-rep(a,b,c,dens,x,atoms),5))
    print("%-14s a,b,c from table; f(1)=%s f'(1)=%s ; residuals at x=.3,2.5,7:"%(name,mp.nstr(f(mp.mpf(1)),4),mp.nstr(mp.diff(f,1),4)),errs)
