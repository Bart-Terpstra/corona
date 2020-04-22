import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
sns.set()  # set Seaborn defaults
plt.rcParams['figure.figsize'] = 8, 6  # default hor./vert. size of plots, in inches
plt.rcParams['lines.markeredgewidth'] = 1  # to fix issue with seaborn box plots; needed after import seaborn

import numpy as np
from tqdm.notebook import tqdm

# Log file to keep track of output
log = open("log.txt",'w') # 'a' will append

OD = np.array([(100,200,300), (400,500,600), (700,800,900)])
thresh = 1000       # determines where the outbreak starts
days = 50


# Initialize the population vector from the origin-destination flow matrix
N_k = np.abs(np.diagonal(OD) + OD.sum(axis=0) - OD.sum(axis=1))
log.write('1.1) --- Diagonal is starting in the location: '+str(np.diagonal(OD))+'\n')
log.write('1.2) --- Axis 0 is total incoming travellers: '+str(OD.sum(axis=0))+'\n')
log.write('1.3) --- Axis 1 is total outgoing travellers '+str(OD.sum(axis=1))+'\n\n')
log.write('1.4) --- Susceptible initialization vector N_k: '+str(N_k)+'\n\n')

# Number of locations
locs_len = len(N_k)
# Make a numpy array with 3 columns for keeping track of the S, I, R groups
SIR = np.zeros(shape=(locs_len, 3))
 # Initialize the S group with the respective populations
SIR[:,0] = N_k
log.write('1.5) --- Initialization of S group in SIR: '+'\n'+str(SIR)+'\n\n')

# For demo purposes, randomly introduce infections depending on the threshold
# Where the number of Susceptible persons is <= threshold, take that number and floor divide by 20, else 0.
first_infections = np.where(SIR[:, 0]<=thresh, SIR[:, 0]//20, 0)
log.write('2.1) --- First infections: '+str(first_infections)+'\n\n')
# Update your Susceptible column
SIR[:, 0] = SIR[:, 0] - first_infections
# Update the Infected column
SIR[:, 1] = SIR[:, 1] + first_infections
# print('Updated with first infections: ', '\n', SIR, '\n')
log.write('2.2) --- SIR matrix with updated with first infections: '+'\n'+str(SIR)+'\n\n')

# Row normalize the SIR matrix to keep track of proportions (values 0-1)
row_sums = SIR.sum(axis=1)
# log.write('Row sums: '+str(SIR)+'\n\n')
SIR_n = SIR / row_sums[:, np.newaxis]
log.write('3.) --- Normalized SIR values: '+'\n'+str(SIR_n)+'\n\n')

# Lists for plotting purposes
susceptible_pop_norm = [SIR_n[0][0]]
infected_pop_norm = [SIR_n[0][1]]
recovered_pop_norm = [SIR_n[0][2]]
t = [0] # time in days

# Initialize parameters
beta = 1.6          # Transmission rate
gamma = 0.04        # Recovery rate
public_trans = 0.5  # Infection rate (= Alpha)
R0 = beta/gamma     # Basic reproduction number
# Create vectors of beta and gamma values
# Random numbers from Gamma distribution to ensure a random R0 at each location
# beta_vec = np.random.gamma(1.6, 2, locs_len)
beta_vec = np.full(locs_len, beta) # Use this to make all constant instead
log.write('4.1) --- Beta vector: '+str(beta_vec)+'\n')
gamma_vec = np.full(locs_len, gamma)
log.write('4.2) --- Gamma vector: '+str(gamma_vec)+'\n')
public_trans_vec = np.full(locs_len, public_trans)
log.write('4.3) --- Public transport vector (Alpha): '+str(public_trans_vec)+'\n\n')

# Make copy of the SIR matrices
SIR_sim = SIR.copy()
SIR_nsim = SIR_n.copy()
log.write('5.) --- Normalized simulation matrix: '+'\n'+str(SIR_nsim)+'\n\n')

# Run model
# Checks whether the simulation numbers sum is equal to the initialized numbers sum
# Why?
# print(SIR_sim.sum(axis=0).sum() == N_k.sum()) # 1500 in this case
# print(SIR_sim.sum(axis=0).sum())

log.write('>>>>>> START ITERATING <<<<<<\n\n')
for time_step in tqdm(range(days)):
    print('\n', '-------------------------------> ', time_step)
    log.write('-------------------------------> '+str(time_step)+'\n\n')
    t.append(time_step+1)

    # I think these are the infections due to travelling
    # Gives a matrix of the fraction of infections compared to the total.
    infected_mat = np.array([SIR_nsim[:,1],]*locs_len).transpose()
    log.write('6.1) --- Fraction of infections in OD matrix: '+'\n'+str(infected_mat)+'\n\n')
    # Calculates the number of new infections due to travelling.
    OD_infected = np.round(OD*infected_mat)
    log.write('6.2) --- Number of infections in OD matrix: '+'\n'+str(OD_infected)+'\n\n')
    # Calculates the number of new infections due to travelling.
    inflow_infected = OD_infected.sum(axis=0)
    log.write('6.3) --- Inflow vector summed over rows: '+str(inflow_infected)+'\n')
    inflow_infected = np.round(inflow_infected*public_trans_vec)
    log.write('6.4) --- Infected inflow vector summed over rows: '+str(inflow_infected)+'\n')
    log.write('6.5) --- Total infected inflow sum: '+str(inflow_infected.sum())+'\n\n')

    # Calculates the newly infected persons
    # This is the second equation.
    new_infect = beta_vec*SIR_sim[:, 0]*inflow_infected/(N_k + OD.sum(axis=0))
    log.write('6.6) --- Beta vector:  '+str(beta_vec)+'\n')
    log.write('6.7) --- Susceptible vector:  '+str(SIR_sim[:, 0])+'\n')
    log.write('6.8) --- Inflow infected vector:  '+str(inflow_infected)+'\n')
    log.write('6.9) --- Susceptible initialization vector N_k:  '+str(N_k)+'\n')
    log.write('6.10) --- Incoming travelers vector:  '+str(OD.sum(axis=0))+'\n')
    log.write('6.11) --- New infections vector:  '+str(new_infect)+'\n\n')

    # Calculates the newly recovered persons
    new_recovered = gamma_vec*SIR_sim[:, 1]
    log.write('7.) --- New recovers vector:  '+str(new_recovered)+'\n\n')

    # Update the columns S, I, R
    # First and last line prevent impossible (negative) values
    new_infect = np.where(new_infect>SIR_sim[:, 0], SIR_sim[:, 0], new_infect)
    SIR_sim[:, 0] = SIR_sim[:, 0] - new_infect
    SIR_sim[:, 1] = SIR_sim[:, 1] + new_infect - new_recovered
    SIR_sim[:, 2] = SIR_sim[:, 2] + new_recovered
    SIR_sim = np.where(SIR_sim<0,0,SIR_sim)

    # Recompute the normalized SIR matrix
    row_sums = SIR_sim.sum(axis=1)
    SIR_nsim = SIR_sim / row_sums[:, np.newaxis]

    # Compute the S, I, R fractions
    S = SIR_sim[:,0].sum()/N_k.sum()
    I = SIR_sim[:,1].sum()/N_k.sum()
    R = SIR_sim[:,2].sum()/N_k.sum()
    # The last two values must be equal
    print(round(S,4), round(I,4), round(R,4), (S+I+R)*N_k.sum(), N_k.sum())

    # Lists with fractions of I, S, R for plotting purposes
    susceptible_pop_norm.append(S)
    infected_pop_norm.append(I)
    recovered_pop_norm.append(R)

print('SIR matrix at initialization: \n',SIR, '\n')
log.write('8.1) --- SIR matrix at initialization: '+'\n'+str(SIR)+'\n\n')
print('Final rounded SIR matrix: \n',SIR_sim.round(1))
log.write('8.2) --- Final rounded SIR matrix: \n'+str(SIR_sim.round(1)))

# Close the log
log.close()

# Plotting
fig = plt.figure()
to_plot = np.stack([susceptible_pop_norm, infected_pop_norm, recovered_pop_norm]).T
plt.plot(t, to_plot)
plt.ylabel('Population fraction')
plt.xlabel('Time (days)')
plt.legend(['Susceptible', 'Infected', 'Recovered'])
plt.title(r'SIR Model with $a={}$, $b={}$, $y={}$'.format(public_trans, beta, gamma))
plt.show()
