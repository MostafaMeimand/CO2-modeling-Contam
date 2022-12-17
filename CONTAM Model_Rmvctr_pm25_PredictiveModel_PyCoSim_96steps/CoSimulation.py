# Contam CoSimulation
import pandas as pd
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#%%
def Text_Generator():

    temporary_text = ""
    for i in range(96):
        temporary_text += str(Actions['hour'][i]) + ":" + str(Actions['minutes'][i]) + ":00 " + str(Actions["actions"][i]) + " \n"

    return temporary_text

def CoSimulation():

    text = open("fatima@@.prj").read()

    NextFile = open("fatima_updated.prj","wt")
    NextFile.write(text[:3120] + '\n' + Text_Generator() + '\n' + text[4278:])
    NextFile.close()

    os.system("contamx3 fatima_updated")
    os.system("simread fatima_updated < responses.txt")

def Dataset_Cleaning():   
    return pd.read_table("fatima_updated.ncr")['1'] * 400/0.000608

#%%
Actions = pd.DataFrame(range(96))
Actions["index"] = Actions.index
Actions["minutes"] = Actions["index"] % 4 * 15
Actions["minutes"][Actions["minutes"] == 0] = '00'
temp = []
for i in range(0,24):
    temp.append([i] * 4)
Actions["hour"] = np.reshape(temp,(1,96))[0]
#%% a simple PID controller
Actions["actions"] = 0
CoSimulation()
while (Dataset_Cleaning() > 1000).sum() != 1:
    CoSimulation()
    Actions["actions"][Dataset_Cleaning()[Dataset_Cleaning() > 1000].index[0] - 1] = 1
    CoSimulation()
#%%
plt.plot(Dataset_Cleaning(), linewidth = 2)
plt.xticks(range(0,100,4),range(25))
plt.xlabel("Hour")
plt.ylabel("Co2 concentration (ppm)")
plt.axhline(y = 400, color = '#FF7F24', ls = "--", linewidth = 2)
plt.axhline(y = 1000, color = '#FF7F24', ls = "--", linewidth = 2)
# plt.savefig("PID contorller.svg")

#%% plotting based on differnt On/Off intervals
Actions["actions"] = 0
for i in range(96):
    if i % 4 == 0:
        Actions["actions"][i] = 1

CoSimulation()
Energy_60 = Dataset_Cleaning()
plt.plot(Every_60, linewidth = 2)
# plt.plot(Every_90, linewidth = 2)
# plt.plot(Every_120, linewidth = 2)
plt.xticks(range(0,100,4),range(25))
plt.xlabel("Hour")
plt.ylabel("Co2 concentration (ppm)")
plt.axhline(y = 400, color = '#FF7F24', ls = "--", linewidth = 2)
plt.axhline(y = 1000, color = '#FF7F24', ls = "--", linewidth = 2)
plt.legend(["60'","90'","120'"])
# plt.savefig("Co2Concentration - hour.svg")




