import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
sns.set()  # set Seaborn defaults
plt.rcParams['figure.figsize'] = 8, 6  # default hor./vert. size of plots, in inches
plt.rcParams['lines.markeredgewidth'] = 1  # to fix issue with seaborn box plots; needed after import seaborn

import numpy as np
from tqdm.notebook import tqdm
import pandas as pd

# Log file to keep track of output
log = open("log_62_munies.txt",'w') # 'a' will append

# Read the OD_w matrix of the 62 municipalities
OD_w = pd.read_excel(r'C:\Users\20195526\Documents\PDEng Angeliki\Projects\Case Studies\2. Corona Virus Module/OD_matrix_working.xlsx', index_col='origin')
OD_w = OD_w.to_numpy()
working_pop_pm_vec = np.sum(OD_w, axis = 1)
thresh_w = np.sum(OD_w, axis = 1).max() #150000#85000#1000       # determines where the outbreak starts
days = 100 #50
floor_division = 2000 #When 2000, it means that 0.05% of the pop gets infected at first. The bigger this number is the later the peak of the infection curve will start because the initial population that was infected was very small.

# Initialize the population vector from the origin-destination flow matrix
N_k_w = np.abs(np.diagonal(OD_w) + OD_w.sum(axis=0) - OD_w.sum(axis=1))
log.write('1.1) --- Diagonal is starting in the location: '+str(np.diagonal(OD_w))+'\n')
log.write('1.2) --- Axis 0 is total incoming travellers: '+str(OD_w.sum(axis=0))+'\n')
log.write('1.3) --- Axis 1 is total outgoing travellers '+str(OD_w.sum(axis=1))+'\n\n')
log.write('1.4) --- Susceptible initialization vector N_k_w: '+str(N_k_w)+'\n\n')

# Number of locations
locs_len_w = len(N_k_w)
# Make a numpy array with 4 columns for keeping track of the S, I, R, D groups
SIR_W = np.zeros(shape=(locs_len_w, 4))
 # Initialize the S group with the respective populations
SIR_W[:,0] = N_k_w
log.write('1.5) --- Initialization of S group in SIR_W: '+'\n'+str(SIR_W)+'\n\n')

# For demo purposes, randomly intrOD_wuce infections depending on the thresh_wold
# Where the number of Susceptible persons is <= thresh_w old, take that number and floor divide by 20, else 0.
first_infections_w = np.where(SIR_W[:, 0]<=thresh_w, SIR_W[:, 0]//floor_division, 0)
log.write('2.1) --- First infections: '+str(first_infections_w)+'\n\n')
# Update your Susceptible column
SIR_W[:, 0] = SIR_W[:, 0] - first_infections_w
# Update the Infected column
SIR_W[:, 1] = SIR_W[:, 1] + first_infections_w
# print('Updated with first infections: ', '\n', SIR_W, '\n')
log.write('2.2) --- SIR_W matrix with updated with first infections: '+'\n'+str(SIR_W)+'\n\n')

# Row normalize the SIR_W matrix to keep track of proportions (values 0-1)
row_sums_w = SIR_W.sum(axis=1)
# log.write('Row sums: '+str(SIR_W)+'\n\n')
SIR_W_n = SIR_W / row_sums_w[:, np.newaxis]
log.write('3.) --- Normalized SIR_W values: '+'\n'+str(SIR_W_n)+'\n\n')

# Lists for plotting purposes
susceptible_pop_norm_w = [SIR_W_n[0][0]]
infected_pop_norm_w = [SIR_W_n[0][1]]
recovered_pop_norm_w = [SIR_W_n[0][2]]
dead_pop_norm_w = [SIR_W_n[0][3]]
t = [0] # time in days

# Initialize parameters
beta_w = 1.6          # Transmission rate (we can tune this parameter based on the actual numbers)
gamma_w = 0.06        # Recovery rate (1/D where D is the number of days a person is infected)
public_trans_w = 0.7  # Infection rate (= Alpha) I think is the percentage of people transfering the disease because they are travelling with e.g public transport. So, when do we assume that the disease is transferred? Only when people use public transport?
R0_w = beta_w/gamma_w     # Basic reprOD_wuction number
dr_w = 0.016       # Death rate
# Create vectors of beta_w and gamma_w values
# Random numbers from gamma_w distribution to ensure a random R0_w at each location
# beta_w_vec = np.random.gamma_w(1.6, 2, locs_len_w)
beta_w_vec = np.full(locs_len_w, beta_w) # Use this to make all constant instead
log.write('4.1) --- beta_w vector: '+str(beta_w_vec)+'\n')
gamma_w_vec = np.full(locs_len_w, gamma_w)
log.write('4.2) --- gamma_w vector: '+str(gamma_w_vec)+'\n')
public_trans_w_vec = np.full(locs_len_w, public_trans_w)
log.write('4.3) --- Public transport vector (Alpha): '+str(public_trans_w_vec)+'\n\n')
dr_w_vec = np.full(locs_len_w, dr_w)
log.write('4.3) --- Death Rate vector (dr_w): '+str(dr_w_vec)+'\n\n')

# Make copy of the SIR_W matrices
SIR_W_sim = SIR_W.copy()
SIR_W_nsim = SIR_W_n.copy()
log.write('5.) --- Normalized simulation matrix: '+'\n'+str(SIR_W_nsim)+'\n\n')

# lists for map plotting purposes
map_I_w = np.array(SIR_W_nsim[:,1])
map_I_w = map_I_w.reshape(62,1)

# Run mOD_wel
# Checks whether the simulation numbers sum is equal to the initialized numbers sum
# Why?
# print(SIR_W_sim.sum(axis=0).sum() == N_k_w.sum()) # 1500 in this case
# print(SIR_W_sim.sum(axis=0).sum())
log.write('>>>>>> START ITERATING <<<<<<\n\n')
for time_step in tqdm(range(days)):
    print('\n', '-------------------------------> ', time_step)
    log.write('-------------------------------> '+str(time_step)+'\n\n')
    t.append(time_step+1)

    # I think these are the infections due to travelling
    # Gives a matrix of the fraction of infections compared to the total.
    infected_mat_w = np.array([SIR_W_nsim[:,1],]*locs_len_w).transpose()
    log.write('6.1) --- Fraction of infections in OD_w matrix: '+'\n'+str(infected_mat_w)+'\n\n')
    # Calculates the number of new infections due to travelling.
    OD_w_infected = np.round(OD_w*infected_mat_w)
    log.write('6.2) --- Number of infections in OD_w matrix: '+'\n'+str(OD_w_infected)+'\n\n')
    # Calculates the number of new infections due to travelling.
    inflow_infected_w = OD_w_infected.sum(axis=0)
    log.write('6.3) --- Inflow vector summed over rows: '+str(inflow_infected_w)+'\n')
    inflow_infected_w = np.round(inflow_infected_w*public_trans_w_vec)
    log.write('6.4) --- Infected inflow vector summed over rows: '+str(inflow_infected_w)+'\n')
    log.write('6.5) --- Total infected inflow sum: '+str(inflow_infected_w.sum())+'\n\n')

    # Calculates the newly infected persons
    # This is the second equation.
    new_infect_w = beta_w_vec*SIR_W_sim[:, 0]*inflow_infected_w/(N_k_w + OD_w.sum(axis=0))
    log.write('6.6) --- beta_w vector:  '+str(beta_w_vec)+'\n')
    log.write('6.7) --- Susceptible vector:  '+str(SIR_W_sim[:, 0])+'\n')
    log.write('6.8) --- Inflow infected vector:  '+str(inflow_infected_w)+'\n')
    log.write('6.9) --- Susceptible initialization vector N_k_w:  '+str(N_k_w)+'\n')
    log.write('6.10) --- Incoming travelers vector:  '+str(OD_w.sum(axis=0))+'\n')
    log.write('6.11) --- New infections vector:  '+str(new_infect_w)+'\n\n')

    # Calculates the newly recovered persons
    new_recovered_w = gamma_w_vec*SIR_W_sim[:, 1]
    log.write('7.) --- New recovers vector:  '+str(new_recovered_w)+'\n\n')

    # Calculates the dead persons
    new_dead_w = dr_w * SIR_W_sim[:, 1]
    log.write('7.) --- New dead vector:  '+str(new_dead_w)+'\n\n')

    # Update the columns S, I, R
    # First and last line prevent impossible (negative) values
    new_infect_w = np.where(new_infect_w>SIR_W_sim[:, 0], SIR_W_sim[:, 0], new_infect_w)
    SIR_W_sim[:, 0] = SIR_W_sim[:, 0] - new_infect_w                  # Susceptible
    SIR_W_sim[:, 1] = SIR_W_sim[:, 1] + new_infect_w - new_recovered_w  - new_dead_w # Infected

    SIR_W_sim[:, 2] = SIR_W_sim[:, 2] + new_recovered_w              # Recovered
    SIR_W_sim[:, 3] = SIR_W_sim[:,3] + new_dead_w
    SIR_W_sim = np.where(SIR_W_sim<0,0,SIR_W_sim)

    # Recompute the normalized SIR_W matrix
    row_sums_w = SIR_W_sim.sum(axis=1)
    SIR_W_nsim = SIR_W_sim / row_sums_w[:, np.newaxis]

    # Compute the S, I, R, D fractions
    S_w = SIR_W_sim[:,0].sum()/N_k_w.sum()
    I_w = SIR_W_sim[:,1].sum()/N_k_w.sum()
    R_w = SIR_W_sim[:,2].sum()/N_k_w.sum()
    D_w = SIR_W_sim[:,3].sum()/N_k_w.sum()

    # The last two values must be equal
    print(round(S_w,4), round(I_w,4), round(R_w,4),  round(D_w,4), (S_w+I_w+R_w+D_w)*N_k_w.sum(), N_k_w.sum())
    log.write('temp.) --- Check number equality:  '+str([round(S_w,4), round(I_w,4), round(R_w,4),  round(D_w,4), (S_w+I_w+R_w+D_w)*N_k_w.sum(), N_k_w.sum()])+'\n\n')

    # Lists with fractions of I, S, R for plotting purposes
    susceptible_pop_norm_w.append(S_w)
    infected_pop_norm_w.append(I_w)
    recovered_pop_norm_w.append(R_w)
    dead_pop_norm_w.append(D_w)

    # For map plotting purposes, save the values from each iteration
    # map_I_w.append(SIR_W_sim[:,1]) # infected
    map_I_w = np.append(map_I_w, SIR_W_nsim[:,1].reshape(62,1), axis=1)
    # map_D.append(SIR_W_sim[:,3]) # dead

print('SIR_W matrix at initialization: \n',SIR_W, '\n')
log.write('8.1) --- SIR_W matrix at initialization: '+'\n'+str(SIR_W)+'\n\n')
print('Final rounded SIR_W matrix: \n',SIR_W_sim.round(1))
log.write('8.2) --- Final rounded SIR_W matrix: \n'+str(SIR_W_sim.round(1)))

# Close the log
log.close()

# Create a dataframe
data_w = {'Susceptible': susceptible_pop_norm_w, 'Infected': infected_pop_norm_w, 'Recovered': recovered_pop_norm_w, 'Dead': dead_pop_norm_w}
df_w = pd.DataFrame(data_w)
df_w.to_csv(r'C:\Users\20195526\Documents\PDEng Angeliki\Projects\Case Studies\2. Corona Virus Module/SIR_W_values_62_munies_SIR_W.csv')

# Save the infections per municipality
df_map_I_w = pd.DataFrame(map_I_w)
np.save(r'C:\Users\20195526\Documents\PDEng Angeliki\Projects\Case Studies\2. Corona Virus Module/I_62_101_W', map_I_w)



'''NON-WORKING POPULATION - SAME STORY - DIFFERENT PARAMETERS ALPHA & DEATH RATIO'''

# Log file to keep track of output
log = open("log_62_munies.txt",'w') # 'a' will append

# Read the OD_nw matrix of the 62 municipalities
OD_nw = pd.read_excel(r'C:\Users\20195526\Documents\PDEng Angeliki\Projects\Case Studies\2. Corona Virus Module/OD_matrix_no_working.xlsx', index_col='origin')
OD_nw = OD_nw.to_numpy()
non_working_pop_pm_vec = np.sum(OD_nw, axis = 1)
thresh_nw = np.sum(OD_nw, axis = 1).max() #150000#85000#1000       # determines where the outbreak starts
days = 100 #50
floor_division = 2000 #When 2000, it means that 0.05% of the pop gets infected at first. The bigger this number is the later the peak of the infection curve will start because the initial population that was infected was very small.

# Initialize the population vector from the origin-destination flow matrix
N_k_nw = np.abs(np.diagonal(OD_nw) + OD_nw.sum(axis=0) - OD_nw.sum(axis=1))
log.write('1.1) --- Diagonal is starting in the location: '+str(np.diagonal(OD_nw))+'\n')
log.write('1.2) --- Axis 0 is total incoming travellers: '+str(OD_nw.sum(axis=0))+'\n')
log.write('1.3) --- Axis 1 is total outgoing travellers '+str(OD_nw.sum(axis=1))+'\n\n')
log.write('1.4) --- Susceptible initialization vector N_k_nw: '+str(N_k_nw)+'\n\n')

# Number of locations
locs_len_nw = len(N_k_nw)
# Make a numpy array with 4 columns for keeping track of the S, I, R, D groups
SIR_NW = np.zeros(shape=(locs_len_nw, 4))
 # Initialize the S group with the respective populations
SIR_NW[:,0] = N_k_nw
log.write('1.5) --- Initialization of S group in SIR_NW: '+'\n'+str(SIR_NW)+'\n\n')

# For demo purposes, randomly intrOD_nwuce infections depending on the thresh_nwold
# Where the number of Susceptible persons is <= thresh_nw old, take that number and floor divide by 20, else 0.
first_infections_nw = np.where(SIR_NW[:, 0]<=thresh_nw, SIR_NW[:, 0]//floor_division, 0)
log.write('2.1) --- First infections: '+str(first_infections_nw)+'\n\n')
# Update your Susceptible column
SIR_NW[:, 0] = SIR_NW[:, 0] - first_infections_nw
# Update the Infected column
SIR_NW[:, 1] = SIR_NW[:, 1] + first_infections_nw
# print('Updated with first infections: ', '\n', SIR_NW, '\n')
log.write('2.2) --- SIR_NW matrix with updated with first infections: '+'\n'+str(SIR_NW)+'\n\n')

# Row normalize the SIR_NW matrix to keep track of proportions (values 0-1)
row_sums_nw = SIR_NW.sum(axis=1)
# log.write('Row sums: '+str(SIR_NW)+'\n\n')
SIR_NW_n = SIR_NW / row_sums_nw[:, np.newaxis]
log.write('3.) --- Normalized SIR_NW values: '+'\n'+str(SIR_NW_n)+'\n\n')

# Lists for plotting purposes
susceptible_pop_norm_nw = [SIR_NW_n[0][0]]
infected_pop_norm_nw = [SIR_NW_n[0][1]]
recovered_pop_norm_nw = [SIR_NW_n[0][2]]
dead_pop_norm_nw = [SIR_NW_n[0][3]]
t = [0] # time in days

# Initialize parameters
beta_nw = 1.6          # Transmission rate (we can tune this parameter based on the actual numbers)
gamma_nw = 0.06        # Recovery rate (1/D where D is the number of days a person is infected)
public_trans_nw = 0.5  # Infection rate (= Alpha) I think is the percentage of people transfering the disease because they are travelling with e.g public transport. So, when do we assume that the disease is transferred? Only when people use public transport?
R0_nw = beta_nw/gamma_nw     # Basic reprOD_nwuction number
dr_nw = 0.25       # Death rate
# Create vectors of beta_nw and gamma_nw values
# Random numbers from gamma_nw distribution to ensure a random R0_nw at each location
# beta_nw_vec = np.random.gamma_nw(1.6, 2, locs_len_nw)
beta_nw_vec = np.full(locs_len_nw, beta_nw) # Use this to make all constant instead
log.write('4.1) --- beta_nw vector: '+str(beta_nw_vec)+'\n')
gamma_nw_vec = np.full(locs_len_nw, gamma_nw)
log.write('4.2) --- gamma_nw vector: '+str(gamma_nw_vec)+'\n')
public_trans_nw_vec = np.full(locs_len_nw, public_trans_nw)
log.write('4.3) --- Public transport vector (Alpha): '+str(public_trans_nw_vec)+'\n\n')
dr_nw_vec = np.full(locs_len_nw, dr_nw)
log.write('4.3) --- Death Rate vector (dr_nw): '+str(dr_nw_vec)+'\n\n')

# Make copy of the SIR_NW matrices
SIR_NW_sim = SIR_NW.copy()
SIR_NW_nsim = SIR_NW_n.copy()
log.write('5.) --- Normalized simulation matrix: '+'\n'+str(SIR_NW_nsim)+'\n\n')

# lists for map plotting purposes
map_I_nw = np.array(SIR_NW_nsim[:,1])
map_I_nw = map_I_nw.reshape(62,1)

# Run mOD_nwel
# Checks whether the simulation numbers sum is equal to the initialized numbers sum
# Why?
# print(SIR_NW_sim.sum(axis=0).sum() == N_k_nw.sum()) # 1500 in this case
# print(SIR_NW_sim.sum(axis=0).sum())
log.write('>>>>>> START ITERATING <<<<<<\n\n')
for time_step in tqdm(range(days)):
    print('\n', '-------------------------------> ', time_step)
    log.write('-------------------------------> '+str(time_step)+'\n\n')
    t.append(time_step+1)

    # I think these are the infections due to travelling
    # Gives a matrix of the fraction of infections compared to the total.
    infected_mat_nw = np.array([SIR_NW_nsim[:,1],]*locs_len_nw).transpose()
    log.write('6.1) --- Fraction of infections in OD_nw matrix: '+'\n'+str(infected_mat_nw)+'\n\n')
    # Calculates the number of new infections due to travelling.
    OD_nw_infected = np.round(OD_nw*infected_mat_nw)
    log.write('6.2) --- Number of infections in OD_nw matrix: '+'\n'+str(OD_nw_infected)+'\n\n')
    # Calculates the number of new infections due to travelling.
    inflow_infected_nw = OD_nw_infected.sum(axis=0)
    log.write('6.3) --- Inflow vector summed over rows: '+str(inflow_infected_nw)+'\n')
    inflow_infected_nw = np.round(inflow_infected_nw*public_trans_nw_vec)
    log.write('6.4) --- Infected inflow vector summed over rows: '+str(inflow_infected_nw)+'\n')
    log.write('6.5) --- Total infected inflow sum: '+str(inflow_infected_nw.sum())+'\n\n')

    # Calculates the newly infected persons
    # This is the second equation.
    new_infect_nw = beta_nw_vec*SIR_NW_sim[:, 0]*inflow_infected_nw/(N_k_nw + OD_nw.sum(axis=0))
    log.write('6.6) --- beta_nw vector:  '+str(beta_nw_vec)+'\n')
    log.write('6.7) --- Susceptible vector:  '+str(SIR_NW_sim[:, 0])+'\n')
    log.write('6.8) --- Inflow infected vector:  '+str(inflow_infected_nw)+'\n')
    log.write('6.9) --- Susceptible initialization vector N_k_nw:  '+str(N_k_nw)+'\n')
    log.write('6.10) --- Incoming travelers vector:  '+str(OD_nw.sum(axis=0))+'\n')
    log.write('6.11) --- New infections vector:  '+str(new_infect_nw)+'\n\n')

    # Calculates the newly recovered persons
    new_recovered_nw = gamma_nw_vec*SIR_NW_sim[:, 1]
    log.write('7.) --- New recovers vector:  '+str(new_recovered_nw)+'\n\n')

    # Calculates the dead persons
    new_dead_nw = dr_nw * SIR_NW_sim[:, 1]
    log.write('7.) --- New dead vector:  '+str(new_dead_nw)+'\n\n')

    # Update the columns S, I, R
    # First and last line prevent impossible (negative) values
    new_infect_nw = np.where(new_infect_nw>SIR_NW_sim[:, 0], SIR_NW_sim[:, 0], new_infect_nw)
    SIR_NW_sim[:, 0] = SIR_NW_sim[:, 0] - new_infect_nw                  # Susceptible
    SIR_NW_sim[:, 1] = SIR_NW_sim[:, 1] + new_infect_nw - new_recovered_nw  - new_dead_nw # Infected

    SIR_NW_sim[:, 2] = SIR_NW_sim[:, 2] + new_recovered_nw              # Recovered
    SIR_NW_sim[:, 3] = SIR_NW_sim[:,3] + new_dead_nw
    SIR_NW_sim = np.where(SIR_NW_sim<0,0,SIR_NW_sim)

    # Recompute the normalized SIR_NW matrix
    row_sums_nw = SIR_NW_sim.sum(axis=1)
    SIR_NW_nsim = SIR_NW_sim / row_sums_nw[:, np.newaxis]

    # Compute the S, I, R, D fractions
    S_nw = SIR_NW_sim[:,0].sum()/N_k_nw.sum()
    I_nw = SIR_NW_sim[:,1].sum()/N_k_nw.sum()
    R_nw = SIR_NW_sim[:,2].sum()/N_k_nw.sum()
    D_nw = SIR_NW_sim[:,3].sum()/N_k_nw.sum()

    # The last two values must be equal
    print(round(S_nw,4), round(I_nw,4), round(R_nw,4),  round(D_nw,4), (S_nw+I_nw+R_nw+D_nw)*N_k_nw.sum(), N_k_nw.sum())
    log.write('temp.) --- Check number equality:  '+str([round(S_nw,4), round(I_nw,4), round(R_nw,4),  round(D_nw,4), (S_nw+I_nw+R_nw+D_nw)*N_k_nw.sum(), N_k_nw.sum()])+'\n\n')

    # Lists with fractions of I, S, R for plotting purposes
    susceptible_pop_norm_nw.append(S_nw)
    infected_pop_norm_nw.append(I_nw)
    recovered_pop_norm_nw.append(R_nw)
    dead_pop_norm_nw.append(D_nw)

    # For map plotting purposes, save the values from each iteration
    # map_I_nw.append(SIR_NW_sim[:,1]) # infected
    map_I_nw = np.append(map_I_nw, SIR_NW_nsim[:,1].reshape(62,1), axis=1)
    # map_D.append(SIR_NW_sim[:,3]) # dead

print('SIR_NW matrix at initialization: \n',SIR_NW, '\n')
log.write('8.1) --- SIR_NW matrix at initialization: '+'\n'+str(SIR_NW)+'\n\n')
print('Final rounded SIR_NW matrix: \n',SIR_NW_sim.round(1))
log.write('8.2) --- Final rounded SIR_NW matrix: \n'+str(SIR_NW_sim.round(1)))

# Close the log
log.close()

# Create a dataframe
data_nw = {'Susceptible': susceptible_pop_norm_nw, 'Infected': infected_pop_norm_nw, 'Recovered': recovered_pop_norm_nw, 'Dead': dead_pop_norm_nw}
df_nw = pd.DataFrame(data_nw)
df_nw.to_csv(r'C:\Users\20195526\Documents\PDEng Angeliki\Projects\Case Studies\2. Corona Virus Module/SIR_NW_values_62_munies_SIR_NW.csv')

# Save the infections per municipality
df_map_I_nw = pd.DataFrame(map_I_nw)
np.save(r'C:\Users\20195526\Documents\PDEng Angeliki\Projects\Case Studies\2. Corona Virus Module/I_62_101_NW', map_I_nw)

# Combined DataFrames of Working & Non_Working
df_w_pop = df_w*working_pop_pm_vec[0]
df_nw_pop = df_nw*non_working_pop_pm_vec[0]
df_all = (df_w_pop + df_nw_pop) / (working_pop_pm_vec[0] + non_working_pop_pm_vec[0])
df_all.to_csv(r'C:\Users\20195526\Documents\PDEng Angeliki\Projects\Case Studies\2. Corona Virus Module/SIR_ALL_values_62_munies_SIR_ALL.csv')

# Combine INFECTED working + non_working population FOR THE MAP
working_pop_pm = np.tile(working_pop_pm_vec,(101,1)).T
non_working_pop_pm = np.tile(non_working_pop_pm_vec,(101,1)).T
total_population= working_pop_pm + non_working_pop_pm

map_I = (map_I_w*working_pop_pm + map_I_nw*non_working_pop_pm) / total_population
np.save(r'C:\Users\20195526\Documents\PDEng Angeliki\Projects\Case Studies\2. Corona Virus Module/I_62_101_COMBINED', map_I_nw)

# Plotting
fig = plt.figure()
to_plot = np.stack([susceptible_pop_norm_w, infected_pop_norm_w, recovered_pop_norm_w, dead_pop_norm_w]).T
plt.plot(t, to_plot)
plt.ylabel('Population fraction')
plt.xlabel('Time (days)')
plt.legend([ "Susceptible",'Infected', 'Recovered', 'Dead'])
plt.title(r'SIR_W MOD_wel with $a={}$, $b={}$, $y={}$, $dr_w={}$'.format(public_trans_w, beta_w, gamma_w, dr_w))
plt.show()

fig_nw = plt.figure()
to_plot_nw = np.stack([susceptible_pop_norm_nw, infected_pop_norm_nw, recovered_pop_norm_nw, dead_pop_norm_nw]).T
plt.plot(t, to_plot_nw)
plt.ylabel('Population fraction')
plt.xlabel('Time (days)')
plt.legend([ "Susceptible",'Infected', 'Recovered', 'Dead'])
plt.title(r'SIR_NW Model with $a={}$, $b={}$, $y={}$, $dr_nw={}$'.format(public_trans_nw, beta_nw, gamma_nw, dr_nw))
plt.show()

fig_all = plt.figure()
to_plot_all = np.stack([df_all['Susceptible'], df_all['Infected'], df_all['Recovered'], df_all['Dead']]).T
plt.plot(t, to_plot_all)
plt.ylabel('Population fraction')
plt.xlabel('Time (days)')
plt.legend([ "Susceptible",'Infected', 'Recovered', 'Dead'])
plt.title(r'SIR_ALL Model')
# with $a={}$, $b={}$, $y={}$, $dr_nw={}$'.format(public_trans_nw, beta_nw, gamma_nw, dr_nw))
plt.show()