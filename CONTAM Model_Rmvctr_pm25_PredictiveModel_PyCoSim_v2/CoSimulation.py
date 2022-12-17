# Contam CoSimulation
import pandas as pd
import os
import matplotlib.pyplot as plt
import re
#%%
def Text_Generator():
    temporary_text = ""
    for i in range(25):
        temporary_text += str(i) + ":00:00 " + str(Actions["actions"][i]) + "\n"
    return temporary_text

def CoSimulation():
    text = open("fatima_Main.prj").read()
    NextFile = open("fatima_second.prj","wt")
    NextFile.write(text[:3120] + Text_Generator() + text[3414:])
    NextFile.close()
    os.system("contamx3 fatima_second")
    os.system("simread fatima_second < responses.txt")

def Dataset_Cleaning():   
    return pd.read_table("fatima_second.ncr")['1'] * 400/0.0006
#%%
Actions = pd.DataFrame(range(25))
Actions.columns = ["time"]

Actions["actions"] = 0
for i in range(25):
    if i % 2 == 0:
        Actions["actions"][i] = 1
Actions["actions"][10] = 0
# Actions["actions"][0] = 0

CoSimulation()

plt.plot(Dataset_Cleaning())
#%%
print(Dataset_Cleaning()[24])
#%%

timedelta(hours = 0)


