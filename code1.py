import scipy.special as ss
import scipy.integrate as integrate
import numpy as np
from functools import lru_cache

p = 0
q = 1
p_prime_range = range(1, 11)
epsilon_by_r = 0.15 # change this value to get the different plots
u_p_q = ss.jn_zeros(p, q)[-1]
lower_bound = ss.jv(p + 1, u_p_q)**2 / 4 - epsilon_by_r**2 / 2

@lru_cache(None)  # cache
def u_p_prime_n(p_prime, n):
    return ss.jn_zeros(p_prime, n)[-1]

def n_term_integrand(p, q, p_prime, n, t):
    u_p_q = ss.jn_zeros(p, q)[-1]
    u_p_prime_n_value = u_p_prime_n(p_prime, n)
    return t * ss.jv(p, u_p_q * t) * ss.jv(p_prime, u_p_prime_n_value * t) / ss.jv(p_prime + 1, u_p_prime_n_value)
    

for p_prime in p_prime_range:
    l = 1
    u_p_prime_list = ss.jn_zeros(p_prime, 1000)

    while True:
        l_term_sum = 0
        integrals = []
        
        # vectorize
        for n in range(1, l):
            n_term_integral, _ = integrate.quad(lambda t: n_term_integrand(p, q, p_prime, n, t), 0, 1)
            integrals.append(n_term_integral)

        l_term_sum = np.sum(np.square(integrals))

        if l_term_sum > lower_bound:
            print(p_prime, l)
            break

        l += 1