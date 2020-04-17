"""
SEIR model according to this link
"""
# https://towardsdatascience.com/social-distancing-to-slow-the-coronavirus-768292f04296

"""
SEIR formulas

1. Change in people susceptible to the disease and is moderated by the number of infected people and their contact with the infected

2. Change in number of exposed people, which grows based on the contact rate and decreases based on the incubation period after which people become infected.

3. Change in number of infected people based on the exposed population and the incubation period.
It decreases based on the infectious period, so the higher γ is, the more quickly people die/recover (R)

4. Change in number of recovered people

5. Constraint that indicates there are no birth/migration effects in the model; we have a fixed population from beginning to end.

1)      S_ = - rho * beta * S * I
2)      E_ = - rho* beta * S * I - alpha * E
3)      I_ = alpha * E - gamma * I
4)      R_ = gamma * I
5)      N = S + E + I + R

alpha:  infection rate (0-1)
beta:   contact rate (persons)
gamma:  recovery rate (0-1)
rho:    social distancing factor (0 everyone, 1 base situation)
"""

import numpy as np

def base_seir_model(init_vals, params, t):
    """
    Function that calculates the SEIR number over time

    Takes into the initial starting values for S, E, I, R.
    params:
        alpha: inverse incubation period
        beta: average contact rate in the population
        gamma: inverse of mean infectious period
    t: time step in days
    """
    # From the starting values
    S_0, E_0, I_0, R_0 = init_vals
    # Stored in lists
    S, E, I, R = [S_0], [E_0], [I_0], [R_0]
    # From parameters
    alpha, beta, gamma = params
    # "Assumes constant time steps"?
    dt = t[1] - t[0]
    # For every time period...
    for _ in t[1:]:
        next_S = S[-1] - (beta*S[-1]*I[-1])*dt
        next_E = E[-1] + (beta*S[-1]*I[-1] - alpha*E[-1])*dt
        next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
    return np.stack([S, E, I, R]).T

def social_distancing_seir_model(init_vals, params, t):
    """
    Function that calculates the SEIR number over time

    Takes into the initial starting values for S, E, I, R.
    params:
        alpha:  inverse incubation period
        beta:   average contact rate in the population
        gamma:  inverse of mean infectious period
        rho:    social distancing factor
    t: time step in days
    """
    # From the starting values
    S_0, E_0, I_0, R_0 = init_vals
    # Stored in lists
    S, E, I, R = [S_0], [E_0], [I_0], [R_0]
    # From parameters
    alpha, beta, gamma, rho = params
    # "Assumes constant time steps"?
    dt = t[1] - t[0]
    # For every time period...
    for _ in t[1:]:
        next_S = S[-1] - (rho*beta*S[-1]*I[-1])*dt
        next_E = E[-1] + (rho*beta*S[-1]*I[-1] - alpha*E[-1])*dt
        next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)
    return np.stack([S, E, I, R]).T

"""
Run the base model
"""
# Define parameters
# Number of days
t_max = 100
# To create 1000 steps? So 10 for each day? To make the curve more smooth?
dt = .1
# np.linspace(start_value, end_value, nr_of_values)
t = np.linspace(0, t_max, int(t_max/dt) + 1)
N = 10000
# Normalized population values
init_vals = 1 - 1/N, 1/N, 0, 0
# Params
alpha = 0.2
beta = 1.75
gamma = 0.5
params = alpha, beta, gamma
# Run simulation
results = base_seir_model(init_vals, params, t)

"""
Run the social distancing model
"""
# Define parameters
# Number of days
t_max = 100
# To create 1000 steps? So 10 for each day? To make the curve more smooth?
dt = .1
# np.linspace(start_value, end_value, nr_of_values)
t = np.linspace(0, t_max, int(t_max/dt) + 1)
N = 10000
# Normalized population values
init_vals = 1 - 1/N, 1/N, 0, 0
# Params
alpha = 0.2
beta = 1.75
gamma = 0.5
rho = 0.5
params = alpha, beta, gamma, rho
# Run simulation
results_social_distancing = social_distancing_seir_model(init_vals, params, t)



"""
Plot the result
"""

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
sns.set()  # set Seaborn defaults
plt.rcParams['figure.figsize'] = 8, 6  # default hor./vert. size of plots, in inches
plt.rcParams['lines.markeredgewidth'] = 1  # to fix issue with seaborn box plots; needed after import seaborn

fig = plt.figure()
# Base
# sns.lineplot(t, results[:,0], label='Susceptible')
# sns.lineplot(t, results[:,1], label='Exposed')
# sns.lineplot(t, results[:,2], label='Infected')
# sns.lineplot(t, results[:,3], label='Recovered')
# Social distancing
# sns.lineplot(t, results_social_distancing[:,0], label='Susceptible')
# sns.lineplot(t, results_social_distancing[:,1], label='Exposed')
# sns.lineplot(t, results_social_distancing[:,2], label='Infected')
# sns.lineplot(t, results_social_distancing[:,3], label='Recovered')
# plt.legend(['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Susceptible', 'Exposed', 'Infected', 'Recovered'])
plt.plot(t, results)
# plt.plot(t, results_social_distancing)
plt.ylabel('Population fraction')
plt.xlabel('Time (days)')
plt.legend(['Susceptible', 'Exposed', 'Infected', 'Recovered'])#, 'Susceptible_SD', 'Exposed_SD', 'Infected_SD', 'Recovered_SD'])
plt.title(r'SEIR Model with $a={}$, $b={}$, $y={}$'.format(params[0], params[1], params[2]))
# plt.title('SEIR model (\u03B1 = {}, \u03B2 = {}, \u03B3	= {})'.format(alpha, beta, gamma))
# # plt.title('SEIR model (alpha = {}, beta = {}, gamma	= {})'.format(alpha, beta, gamma))
plt.show()
