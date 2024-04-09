import numpy as np
import pandas as pd

from numpy.lib.scimath import sqrt as csqrt
def get_l(a,p,L0,A0,ka,Tex,kp,P0):
    Sqrt = csqrt
    l1 = (2 * 2 ** 0.3333333333333333 * (2 * a * A0 * ka * L0 - kp * L0 * p ** 2 - a * L0 * Tex)) / (
                864 * a ** 4 * ka ** 2 * kp * L0 ** 6 * p * P0 + Sqrt(
            746496 * a ** 8 * ka ** 4 * kp ** 2 * L0 ** 12 * p ** 2 * P0 ** 2 - 55296 * a ** 6 * ka ** 3 * L0 ** 9 * (
                        2 * a * A0 * ka * L0 - kp * L0 * p ** 2 - a * L0 * Tex) ** 3)) ** 0.3333333333333333 + (
                864 * a ** 4 * ka ** 2 * kp * L0 ** 6 * p * P0 + Sqrt(
            746496 * a ** 8 * ka ** 4 * kp ** 2 * L0 ** 12 * p ** 2 * P0 ** 2 - 55296 * a ** 6 * ka ** 3 * L0 ** 9 * (
                        2 * a * A0 * ka * L0 - kp * L0 * p ** 2 - a * L0 * Tex) ** 3)) ** 0.3333333333333333 / (
                12. * 2 ** 0.3333333333333333 * a ** 2 * ka * L0 ** 3)
    l2 = -((2**0.3333333333333333*(1 + Sqrt(-3))*(2*a*A0*ka*L0 - kp*L0*p**2 - a*L0*Tex))/(864*a**4*ka**2*kp*L0**6*p*P0 + Sqrt(746496*a**8*ka**4*kp**2*L0**12*p**2*P0**2 - 55296*a**6*ka**3*L0**9*(2*a*A0*ka*L0 - kp*L0*p**2 - a*L0*Tex)**3))**0.3333333333333333) - ((1 - Sqrt(-3))*(864*a**4*ka**2*kp*L0**6*p*P0 + Sqrt(746496*a**8*ka**4*kp**2*L0**12*p**2*P0**2 - 55296*a**6*ka**3*L0**9*(2*a*A0*ka*L0 - kp*L0*p**2 - a*L0*Tex)**3))**0.3333333333333333)/(24.*2**0.3333333333333333*a**2*ka*L0**3)
    l3 = -((2**0.3333333333333333*(1 - Sqrt(-3))*(2*a*A0*ka*L0 - kp*L0*p**2 - a*L0*Tex))/(864*a**4*ka**2*kp*L0**6*p*P0 + Sqrt(746496*a**8*ka**4*kp**2*L0**12*p**2*P0**2 - 55296*a**6*ka**3*L0**9*(2*a*A0*ka*L0 - kp*L0*p**2 - a*L0*Tex)**3))**0.3333333333333333) - ((1 + Sqrt(-3))*(864*a**4*ka**2*kp*L0**6*p*P0 + Sqrt(746496*a**8*ka**4*kp**2*L0**12*p**2*P0**2 - 55296*a**6*ka**3*L0**9*(2*a*A0*ka*L0 - kp*L0*p**2 - a*L0*Tex)**3))**0.3333333333333333)/(24.*2**0.3333333333333333*a**2*ka*L0**3)

    l_vals = np.array([l1,l2,l3])
    l_vals_real = l_vals.real
    is_real = np.abs(l_vals.imag)<1e-14
    all_real = is_real.sum(axis=0)==3
    l_low = (~all_real)*l1 + all_real*(l_vals_real.max(axis=0))

    l_high = (kp*p*P0)/(L0*(a*ka + kp*p**2 + a*Tex))
    l = l_low*(a<A0) + l_high*(a>=A0)
    return l


N_iter = 10000
A0 = np.random.uniform(0.1,4,N_iter)
P0 = np.random.uniform(-10,10,N_iter)
ka = 10.0**np.random.uniform(-3,-0,N_iter)
Tex = - 10.0**np.random.uniform(-3,-0,N_iter)
kp = 10.0**np.random.uniform(-3,-0,N_iter)

l_vals = np.array(get_l(1,3.8,1,A0,ka,Tex,kp,P0))

complex_mask = np.abs(l_vals.imag[:3])>1e-15


df = pd.DataFrame({"A0":A0,"P0":P0,"ka":np.log(ka),"Tex":np.log(-Tex),"kp":np.log(kp),"l":l_vals})
df_accept = df.copy()
df_accept = df_accept[(df_accept["l"]>0.1)*(df_accept["l"]<10)]


# sns.pairplot(data=df)
sns.pairplot(data=df_accept)
plt.show()

print(((l_vals[0].imag)**2).min(),((l_vals[1].imag)**2).min(),((l_vals[2].imag)**2).min())