import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
sns.set()  # set Seaborn defaults
plt.rcParams['figure.figsize'] = 8, 6  # default hor./vert. size of plots, in inches
plt.rcParams['lines.markeredgewidth'] = 1  # to fix issue with seaborn box plots; needed after import seaborn
import pandas as pd
import numpy as np

def sens(group, case, subcase):
    beta =  df_parameters["beta"][(df_parameters["Group"] == group) & (df_parameters["Case"] == case) & (df_parameters["Subcase"] == subcase) ]
    gamma =  df_parameters["gamma"][(df_parameters["Group"] == group) & (df_parameters["Case"] == case) & (df_parameters["Subcase"] == subcase) ]
    public_trans =  df_parameters["public_trans"][(df_parameters["Group"] == group) & (df_parameters["Case"] == case) & (df_parameters["Subcase"] == subcase) ]
    death_rate =  df_parameters["dr_w"][(df_parameters["Group"] == group) & (df_parameters["Case"] == case) & (df_parameters["Subcase"] == subcase) ]
    return beta, gamma, public_trans, death_rate

folder_name = 'D:/04_case_study/corona_13/corona'
df_parameters = pd.read_excel(str(folder_name + '/sir_od_matrix/data/Sensitivity values.xlsx'), sheet_name = 0)
t = list(range(101))

#%% CASE 1 - BETA is changing
for row in range(len(df_parameters.index)):
    gr = df_parameters.loc[row, "Group"]
    ca = df_parameters.loc[row, "Case"]
    sub = df_parameters.loc[row, "Subcase"]
    
    if (ca == "Case1") & (gr == "Working"):
        pd_results = pd.read_excel(str(folder_name + '/sir_od_matrix/data/Sensitivity values.xlsx'), sheet_name = 1)
        beta, gamma, public_trans, death_rate = sens(gr, ca, sub)
        if sub == "Subcase1":
            beta1, gamma_w, public_trans_w, death_rate_w = beta, gamma, public_trans, death_rate
            beta1_1_w = beta1.tolist()
        elif sub == "Subcase2":
            beta2, gamma_w, public_trans_w, death_rate_w = beta, gamma, public_trans, death_rate
            beta1_2_w = beta2.tolist()
        elif sub == "Subcase3":
            beta3, gamma_w, public_trans_w, death_rate_w = beta, gamma, public_trans, death_rate
            beta1_3_w = beta3.tolist()
            gamma_w = gamma_w.tolist()
            public_trans_w = public_trans_w.tolist()
            death_rate_w = death_rate_w.tolist()
            
    if (ca == "Case1") & (gr == "Non-working"):
        pd_results = pd.read_excel(str(folder_name + '/sir_od_matrix/data/Sensitivity values.xlsx'), sheet_name = 1)
        beta, gamma, public_trans, death_rate = sens(gr, ca, sub)
        if sub == "Subcase1":
            beta1, gamma_nw, public_trans_nw, death_rate_nw = beta, gamma, public_trans, death_rate
            beta1_1_nw = beta1.tolist()
        elif sub == "Subcase2":
            beta2, gamma_nw, public_trans_nw, death_rate_nw = beta, gamma, public_trans, death_rate
            beta1_2_nw = beta2.tolist()
        elif sub == "Subcase3":
            beta3, gamma_nw, public_trans_nw, death_rate_nw = beta, gamma, public_trans, death_rate
            beta1_3_nw = beta3.tolist()
            gamma_nw = gamma_nw.tolist()
            public_trans_nw = public_trans_nw.tolist()
            death_rate_nw = death_rate_nw.tolist()

#PLOT CASE 1  - Working      
fig_case1_working = plt.figure()
ax = fig_case1_working.add_subplot(1, 1, 1)
ax = fig_case1_working.gca()

ax.set_facecolor('xkcd:white')
ax.spines["bottom"].set_color('black')
ax.spines["left"].set_color('black')
ax.spines["top"].set_color('black')
ax.spines["right"].set_color('black')

ax.grid(color='grey', linestyle='-', linewidth=0.2)
ax.set_xticks(np.arange(0, 120, 20))
ax.set_yticks(np.arange(0, 0.09, 0.01))

to_plot = np.stack([pd_results["W_1_1_Infected"], pd_results["W_1_2_Infected"], pd_results["W_1_3_Infected"], 
                            pd_results["W_1_1_Dead"], pd_results["W_1_2_Dead"], pd_results["W_1_3_Dead"]]).T
plt.plot(t, to_plot)# , label = "Infected-beta=%.2f" %(beta1))
plt.ylabel('Population fraction')
plt.xlabel('Time (days)')
plt.ylim((0,0.08))
plt.title(r"Working Population - gamma=${}$, public trans=${}$, death rate=${}$".format(gamma_w[0], public_trans_w[0], death_rate_w[0]))
plt.legend(["I-$beta = {}$".format(beta1_1_w[0]), "I-$beta = {}$".format(beta1_2_w[0]), "I-$beta = {}$".format(beta1_3_w[0]),
            "D-$beta = {}$".format(beta1_1_w[0]), "D-$beta = {}$".format(beta1_2_w[0]), "D-$beta = {}$".format(beta1_3_w[0])])
plt.savefig('beta_working.png')
plt.show()


#PLOT CASE 1  - Non Working      
fig_case1_non_working = plt.figure()
ax = fig_case1_non_working.add_subplot(1, 1, 1)
ax = fig_case1_non_working.gca()

ax.set_facecolor('xkcd:white')
ax.spines["bottom"].set_color('black')
ax.spines["left"].set_color('black')
ax.spines["top"].set_color('black')
ax.spines["right"].set_color('black')

ax.grid(color='grey', linestyle='-', linewidth=0.2)
ax.set_xticks(np.arange(0, 120, 20))
ax.set_yticks(np.arange(0, 0.14, 0.02))

to_plot = np.stack([pd_results["NW_1_1_Infected"], pd_results["NW_1_2_Infected"], pd_results["NW_1_3_Infected"], 
                            pd_results["NW_1_1_Dead"], pd_results["NW_1_2_Dead"], pd_results["NW_1_3_Dead"]]).T
plt.plot(t, to_plot)# , label = "Infected-beta=%.2f" %(beta1))
plt.ylabel('Population fraction')
plt.xlabel('Time (days)')
plt.ylim((0,0.14))
plt.title(r"Non-working Population - gamma=${}$, public trans=${}$, death rate=${}$".format(gamma_nw[0], public_trans_nw[0], death_rate_nw[0]))
plt.legend(["I-$beta = {}$".format(beta1_1_nw[0]), "I-$beta = {}$".format(beta1_2_nw[0]), "I-$beta = {}$".format(beta1_3_nw[0]),
                    "D-$beta = {}$".format(beta1_1_nw[0]), "D-$beta = {}$".format(beta1_2_nw[0]), "D-$beta = {}$".format(beta1_3_nw[0])])

plt.savefig('beta_non_working.png')
plt.show()


# %% CASE 2 - GAMMA is changing
for row in range(len(df_parameters.index)):
    gr = df_parameters.loc[row, "Group"]
    ca = df_parameters.loc[row, "Case"]
    sub = df_parameters.loc[row, "Subcase"]
    
    if (ca == "Case2") & (gr == "Working"):
        pd_results = pd.read_excel(str(folder_name + '/sir_od_matrix/data/Sensitivity values.xlsx'), sheet_name = 2)
        beta, gamma, public_trans, death_rate = sens(gr, ca, sub)
        if sub == "Subcase1":
            beta_w, gamma_w1, public_trans_w, death_rate_w = beta, gamma, public_trans, death_rate
            gamma_w1 = gamma_w1.tolist()
        elif sub == "Subcase2":
            beta_w, gamma_w2, public_trans_w, death_rate_w = beta, gamma, public_trans, death_rate
            gamma_w2 = gamma_w2.tolist()
        elif sub == "Subcase3":
            beta_w, gamma_w3, public_trans_w, death_rate_w = beta, gamma, public_trans, death_rate
            beta_w = beta_w.tolist()
            gamma_w3 = gamma_w3.tolist()
            public_trans_w = public_trans_w.tolist()
            death_rate_w = death_rate_w.tolist()
  
    if (ca == "Case2") & (gr == "Non-working"):
        pd_results = pd.read_excel(str(folder_name + '/sir_od_matrix/data/Sensitivity values.xlsx'), sheet_name = 2)
        beta, gamma, public_trans, death_rate = sens(gr, ca, sub)
        if sub == "Subcase1":
            beta_nw, gamma_nw1, public_trans_nw, death_rate_nw = beta, gamma, public_trans, death_rate
            gamma_nw1 = gamma_nw1.tolist()
        elif sub == "Subcase2":
            beta_nw, gamma_nw2, public_trans_nw, death_rate_nw = beta, gamma, public_trans, death_rate
            gamma_nw2 = gamma_nw2.tolist()
        elif sub == "Subcase3":
            beta_nw, gamma_nw3, public_trans_nw, death_rate_nw = beta, gamma, public_trans, death_rate
            beta_nw = beta_nw.tolist()
            gamma_nw3 = gamma_nw3.tolist()
            public_trans_nw = public_trans_nw.tolist()
            death_rate_nw = death_rate_nw.tolist()

#PLOT CASE 2 - Working      
fig_case2_working = plt.figure()
ax = fig_case2_working.add_subplot(1, 1, 1)
ax = fig_case2_working.gca()

ax.set_facecolor('xkcd:white')
ax.spines["bottom"].set_color('black')
ax.spines["left"].set_color('black')
ax.spines["top"].set_color('black')
ax.spines["right"].set_color('black')

ax.grid(color='grey', linestyle='-', linewidth=0.2)
ax.set_xticks(np.arange(0, 120, 20))
ax.set_yticks(np.arange(0, 0.09, 0.01))

to_plot = np.stack([pd_results["W_2_1_Infected"], pd_results["W_2_2_Infected"], pd_results["W_2_3_Infected"], 
                            pd_results["W_2_1_Dead"], pd_results["W_2_2_Dead"], pd_results["W_2_3_Dead"]]).T
plt.plot(t, to_plot)# , label = "Infected-beta=%.2f" %(beta1))
plt.ylabel('Population fraction')
plt.xlabel('Time (days)')
plt.ylim((0,0.08))
plt.title(r"Working Population - beta=${}$, public trans=${}$, death rate=${}$".format(beta_w[0], public_trans_w[0], death_rate_w[0]))
plt.legend(["I-$gamma = {}$".format(gamma_w1[0]), "I-$gamma = {}$".format(gamma_w2[0]), "I-$gamma = {}$".format(gamma_w3[0]),
            "D-$gamma = {}$".format(gamma_w1[0]), "D-$gamma = {}$".format(gamma_w2[0]), "D-$gamma = {}$".format(gamma_w3[0])])

plt.savefig('gamma_working.png')
plt.show()


#PLOT CASE 2  - Non Working      
fig_case2_non_working = plt.figure()
ax = fig_case2_non_working.add_subplot(1, 1, 1)
ax = fig_case2_non_working.gca()

ax.set_facecolor('xkcd:white')
ax.spines["bottom"].set_color('black')
ax.spines["left"].set_color('black')
ax.spines["top"].set_color('black')
ax.spines["right"].set_color('black')

ax.grid(color='grey', linestyle='-', linewidth=0.2)
ax.set_xticks(np.arange(0, 120, 20))
ax.set_yticks(np.arange(0, 0.14, 0.02))

to_plot = np.stack([pd_results["NW_2_1_Infected"], pd_results["NW_2_2_Infected"], pd_results["NW_2_3_Infected"], 
                            pd_results["NW_2_1_Dead"], pd_results["NW_2_2_Dead"], pd_results["NW_2_3_Dead"]]).T
plt.plot(t, to_plot)# , label = "Infected-beta=%.2f" %(beta1))
plt.ylabel('Population fraction')
plt.xlabel('Time (days)')
plt.ylim((0,0.14))
plt.title(r"Non-working Population - beta=${}$, public trans=${}$, death rate=${}$".format(beta_nw[0], public_trans_nw[0], death_rate_nw[0]))
plt.legend(["I-$gamma = {}$".format(gamma_nw1[0]), "I-$gamma = {}$".format(gamma_nw2[0]), "I-$gamma = {}$".format(gamma_nw3[0]),
            "D-$gamma = {}$".format(gamma_nw1[0]), "D-$gamma = {}$".format(gamma_nw2[0]), "D-$gamma = {}$".format(gamma_nw3[0])])

plt.savefig('gamma_non_working.png')
plt.show()


# %% CASE 3 - PUBLIC_TRANS is changing
for row in range(len(df_parameters.index)):
    gr = df_parameters.loc[row, "Group"]
    ca = df_parameters.loc[row, "Case"]
    sub = df_parameters.loc[row, "Subcase"]
    
    if (ca == "Case3") & (gr == "Working"):
        pd_results = pd.read_excel(str(folder_name + '/sir_od_matrix/data/Sensitivity values.xlsx'), sheet_name = 3)
        beta, gamma, public_trans, death_rate = sens(gr, ca, sub)
        if sub == "Subcase1":
            beta_w, gamma_w, public_trans_w1, death_rate_w = beta, gamma, public_trans, death_rate
            public_trans_w1 = public_trans_w1.tolist()
        elif sub == "Subcase2":
            beta_w, gamma_w, public_trans_w2, death_rate_w = beta, gamma, public_trans, death_rate
            public_trans_w2 = public_trans_w2.tolist()
        elif sub == "Subcase3":
            beta_w, gamma_w, public_trans_w3, death_rate_w = beta, gamma, public_trans, death_rate
            beta_w = beta_w.tolist()
            gamma_w = gamma_w.tolist()
            public_trans_w3 = public_trans_w3.tolist()
            death_rate_w = death_rate_w.tolist()
  
    if (ca == "Case3") & (gr == "Non-working"):
        pd_results = pd.read_excel(str(folder_name + '/sir_od_matrix/data/Sensitivity values.xlsx'), sheet_name = 3)
        beta, gamma, public_trans, death_rate = sens(gr, ca, sub)
        if sub == "Subcase1":
            beta_nw, gamma_nw, public_trans_nw1, death_rate_nw = beta, gamma, public_trans, death_rate
            public_trans_nw1 = public_trans_nw1.tolist()
        elif sub == "Subcase2":
            beta_nw, gamma_nw, public_trans_nw2, death_rate_nw = beta, gamma, public_trans, death_rate
            public_trans_nw2 = public_trans_nw2.tolist()
        elif sub == "Subcase3":
            beta_nw, gamma_nw, public_trans_nw3, death_rate_nw = beta, gamma, public_trans, death_rate
            beta_nw = beta_nw.tolist()
            gamma_nw = gamma_nw.tolist()
            public_trans_nw3 = public_trans_nw3.tolist()
            death_rate_nw = death_rate_nw.tolist()

#PLOT CASE 3 - Working      
fig_case3_working = plt.figure()
ax = fig_case3_working.add_subplot(1, 1, 1)
ax = fig_case3_working.gca()

ax.set_facecolor('xkcd:white')
ax.spines["bottom"].set_color('black')
ax.spines["left"].set_color('black')
ax.spines["top"].set_color('black')
ax.spines["right"].set_color('black')

ax.grid(color='grey', linestyle='-', linewidth=0.2)
ax.set_xticks(np.arange(0, 120, 20))
ax.set_yticks(np.arange(0, 0.09, 0.01))

to_plot = np.stack([pd_results["W_3_1_Infected"], pd_results["W_3_2_Infected"], pd_results["W_3_3_Infected"], 
                            pd_results["W_3_1_Dead"], pd_results["W_3_2_Dead"], pd_results["W_3_3_Dead"]]).T
plt.plot(t, to_plot)# , label = "Infected-beta=%.2f" %(beta1))
plt.ylabel('Population fraction')
plt.xlabel('Time (days)')
plt.ylim((0,0.08))
plt.title(r"Working Population - beta=${}$, gamma=${}$, death rate=${}$".format(beta_w[0], gamma_w[0], death_rate_w[0]))
plt.legend(["I-pt = ${}$".format(public_trans_w1[0]), "I-pt = ${}$".format(public_trans_w2[0]), "I-pt = ${}$".format(public_trans_w3[0]),
            "D-pt = ${}$".format(public_trans_w1[0]), "D-pt = ${}$".format(public_trans_w2[0]), "D-pt = ${}$".format(public_trans_w3[0])])

plt.savefig('public_trans_working.png')
plt.show()


#PLOT CASE 3  - Non Working      
fig_case3_non_working = plt.figure()
ax = fig_case3_non_working.add_subplot(1, 1, 1)
ax = fig_case3_non_working.gca()

ax.set_facecolor('xkcd:white')
ax.spines["bottom"].set_color('black')
ax.spines["left"].set_color('black')
ax.spines["top"].set_color('black')
ax.spines["right"].set_color('black')

ax.grid(color='grey', linestyle='-', linewidth=0.2)
ax.set_xticks(np.arange(0, 120, 20))
ax.set_yticks(np.arange(0, 0.14, 0.02))

to_plot = np.stack([pd_results["NW_3_1_Infected"], pd_results["NW_3_2_Infected"], pd_results["NW_3_3_Infected"], 
                            pd_results["NW_3_1_Dead"], pd_results["NW_3_2_Dead"], pd_results["NW_3_3_Dead"]]).T
plt.plot(t, to_plot)# , label = "Infected-beta=%.2f" %(beta1))
plt.ylabel('Population fraction')
plt.xlabel('Time (days)')
plt.ylim((0,0.14))
plt.title(r"Non-working Population - beta=${}$, gamma=${}$, death rate=${}$".format(beta_nw[0], gamma_nw[0], death_rate_nw[0]))
plt.legend(["I-pt = ${}$".format(public_trans_nw1[0]), "I-pt = ${}$".format(public_trans_nw2[0]), "I-pt = ${}$".format(public_trans_nw3[0]),
            "D-pt = ${}$".format(public_trans_nw1[0]), "D-pt = ${}$".format(public_trans_nw2[0]), "D-pt = ${}$".format(public_trans_nw3[0])])

plt.savefig('public_trans_non_working.png')
plt.show()

# %% CASE 4 - PUBLIC_TRANS is changing
for row in range(len(df_parameters.index)):
    gr = df_parameters.loc[row, "Group"]
    ca = df_parameters.loc[row, "Case"]
    sub = df_parameters.loc[row, "Subcase"]
    
    if (ca == "Case4") & (gr == "Working"):
        pd_results = pd.read_excel(str(folder_name + '/sir_od_matrix/data/Sensitivity values.xlsx'), sheet_name = 4)
        beta, gamma, public_trans, death_rate = sens(gr, ca, sub)
        if sub == "Subcase1":
            beta_w, gamma_w, public_trans_w, death_rate_w1 = beta, gamma, public_trans, death_rate
            death_rate_w1 = death_rate_w1.tolist()
        elif sub == "Subcase2":
            beta_w, gamma_w, public_trans_w, death_rate_w2 = beta, gamma, public_trans, death_rate
            death_rate_w2 = death_rate_w2.tolist()
        elif sub == "Subcase3":
            beta_w, gamma_w, public_trans_w, death_rate_w3 = beta, gamma, public_trans, death_rate
            beta_w = beta_w.tolist()
            gamma_w = gamma_w.tolist()
            public_trans_w = public_trans_w.tolist()
            death_rate_w3 = death_rate_w3.tolist()
  
    if (ca == "Case4") & (gr == "Non-working"):
        pd_results = pd.read_excel(str(folder_name + '/sir_od_matrix/data/Sensitivity values.xlsx'), sheet_name = 4)
        beta, gamma, public_trans, death_rate = sens(gr, ca, sub)
        if sub == "Subcase1":
            beta_nw, gamma_nw, public_trans_nw, death_rate_nw1 = beta, gamma, public_trans, death_rate
            death_rate_nw1 = death_rate_nw1.tolist()
        elif sub == "Subcase2":
            beta_nw, gamma_nw, public_trans_nw, death_rate_nw2 = beta, gamma, public_trans, death_rate
            death_rate_nw2 = death_rate_nw2.tolist()
        elif sub == "Subcase3":
            beta_nw, gamma_nw, public_trans_nw, death_rate_nw3 = beta, gamma, public_trans, death_rate
            beta_nw = beta_nw.tolist()
            gamma_nw = gamma_nw.tolist()
            public_trans_nw = public_trans_nw.tolist()
            death_rate_nw3 = death_rate_nw3.tolist()

#PLOT CASE 4 - Working      
fig_case4_working = plt.figure()
ax = fig_case4_working.add_subplot(1, 1, 1)
ax = fig_case4_working.gca()

ax.set_facecolor('xkcd:white')
ax.spines["bottom"].set_color('black')
ax.spines["left"].set_color('black')
ax.spines["top"].set_color('black')
ax.spines["right"].set_color('black')

ax.grid(color='grey', linestyle='-', linewidth=0.2)
ax.set_xticks(np.arange(0, 120, 20))
ax.set_yticks(np.arange(0, 0.09, 0.01))

to_plot = np.stack([pd_results["W_4_1_Infected"], pd_results["W_4_2_Infected"], pd_results["W_4_3_Infected"], 
                            pd_results["W_4_1_Dead"], pd_results["W_4_2_Dead"], pd_results["W_4_3_Dead"]]).T
plt.plot(t, to_plot)# , label = "Infected-beta=%.2f" %(beta1))
plt.ylabel('Population fraction')
plt.xlabel('Time (days)')
plt.ylim((0,0.08))
plt.title(r"Working Population - beta=${}$, gamma=${}$, public trans=${}$".format(beta_w[0], gamma_w[0], public_trans_w[0]))
plt.legend(["I-dr = ${}$".format(death_rate_w1[0]), "I-dr = ${}$".format(death_rate_w2[0]), "I-dr = ${}$".format(death_rate_w3[0]),
            "D-dr = ${}$".format(death_rate_w1[0]), "D-dr = ${}$".format(death_rate_w2[0]), "D-dr = ${}$".format(death_rate_w3[0])])

plt.savefig('dr_working.png')
plt.show()


#PLOT CASE 4  - Non Working      
fig_case4_non_working = plt.figure()
ax = fig_case4_non_working.add_subplot(1, 1, 1)
ax = fig_case4_non_working.gca()

ax.set_facecolor('xkcd:white')
ax.spines["bottom"].set_color('black')
ax.spines["left"].set_color('black')
ax.spines["top"].set_color('black')
ax.spines["right"].set_color('black')

ax.grid(color='grey', linestyle='-', linewidth=0.2)
ax.set_xticks(np.arange(0, 120, 20))
ax.set_yticks(np.arange(0, 0.14, 0.02))

to_plot = np.stack([pd_results["NW_4_1_Infected"], pd_results["NW_4_2_Infected"], pd_results["NW_4_3_Infected"], 
                            pd_results["NW_4_1_Dead"], pd_results["NW_4_2_Dead"], pd_results["NW_4_3_Dead"]]).T
plt.plot(t, to_plot)# , label = "Infected-beta=%.2f" %(beta1))
plt.ylabel('Population fraction')
plt.xlabel('Time (days)')
plt.ylim((0,0.14))
plt.title(r"Non Working Population - beta=${}$, gamma=${}$, public trans=${}$".format(beta_nw[0], gamma_nw[0], public_trans_nw[0]))
plt.legend(["I-dr = ${}$".format(death_rate_nw1[0]), "I-dr = ${}$".format(death_rate_nw2[0]), "I-dr = ${}$".format(death_rate_nw3[0]),
            "D-dr = ${}$".format(death_rate_nw1[0]), "D-dr = ${}$".format(death_rate_nw2[0]), "D-dr = ${}$".format(death_rate_nw3[0])])

plt.savefig('dr_non_working.png')
plt.show()
