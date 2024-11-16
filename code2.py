import scipy.special as ss
import scipy.integrate as integrate
import numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt

@lru_cache(None)  # cache
def u_p_prime_n(p_prime, n):
    return ss.jn_zeros(p_prime, n)[-1]

def n_term_integrand(p, q, p_prime, n, t):
    u_p_q = ss.jn_zeros(p, q)[-1]
    u_p_prime_n_value = u_p_prime_n(p_prime, n)
    return t * ss.jv(p, u_p_q * t) * ss.jv(p_prime, u_p_prime_n_value * t) / ss.jv(p_prime + 1, u_p_prime_n_value)
    
    
def find_l_values(p, q, epsilon_by_r, p_prime_range):
    
    u_p_q = ss.jn_zeros(p, q)[-1]
    lower_bound = ss.jv(p + 1, u_p_q)**2 / 4 - epsilon_by_r**2 / 2

    l_values = []
    
    for p_prime in p_prime_range:
    
        l = 1    
        l_term_sum = 0
    
        while True:
        
            lth_term_integral, _ = integrate.quad(lambda t: n_term_integrand(p, q, p_prime, l, t), 0, 1)
            l_term_sum += lth_term_integral ** 2
    
            if l_term_sum > lower_bound:
                # print(p_prime, l)
                break
    
            l += 1

        l_values.append(l)

    return l_values
    
def linear_regression(x_values, y_values):
    n = len(x_values)
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_xy = sum(x * y for x, y in zip(x_values, y_values))
    sum_x_squared = sum(x ** 2 for x in x_values)
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    y_intercept = (sum_y - slope * sum_x) / n
    return slope, y_intercept


m_values = []
c_values = []

p_prime_range = range(1, 11)
p = 0
q = 1
epsilon_by_r_range = range(1, 37)

for epsilon_by_r_100x in epsilon_by_r_range:
    epsilon_by_r = epsilon_by_r_100x/100
    x_values = list(p_prime_range)
    y_values = find_l_values(p, q, epsilon_by_r, p_prime_range)
    # print(y_values)
    m, c = linear_regression(x_values, y_values)
    # print("slope ", m, "y-intercept", c)
    m_values.append(m)
    c_values.append(c)
    print(epsilon_by_r)

epsilon_by_r_list = [i/100 for i in epsilon_by_r_range]

plt.plot(epsilon_by_r_list, m_values)
plt.xlabel("$\epsilon/R$")
plt.ylabel("$m$")
plt.show()

plt.plot(epsilon_by_r_list, c_values)
plt.xlabel("$\epsilon/R$")
plt.ylabel("$c$")
plt.show()
