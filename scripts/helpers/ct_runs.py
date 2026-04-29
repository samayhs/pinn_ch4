import cantera as ct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers.train_StateNet2 import run_cantera_sim, get_input_data_from_states

def get_states(gas, T, P, phi):
    state_path = f"results/26/states_T={T}K, P={P}atm, phi={phi}.csv"

    print(f"Running Cantera: T={T}K, P={P}atm, phi={phi}")
    t_end = 1e-3
    states, IDT = run_cantera_sim(gas, T, P, phi, t_end)

    i = 0
    # try to have 1000ish states per run, and t_end be approx IDT*5
    while IDT * 5 > t_end:
        i += 1
        t_end = IDT * 5
        states, IDT = run_cantera_sim(gas, T, P, phi, t_end, dt=min(1e-4, t_end/1000))
        print(f"Run {i}: IDT:{IDT:.3g}, t_end:{t_end:.3g}")
    data = get_input_data_from_states(states,IDT, save_csv=True, csv_path=state_path)

def dt_finder(csv_path):
    data = np.loadtxt(csv_path, delimiter=",")
    # print(data.shape)
    t1 = data[0,-2]
    print(t1)
    for row in range(len(data) - 1):
        t_new = data[row+1, -2]
        print(t_new-t1)
        t1 = t_new
        

def plot_csv(csv_paths):
    

    fig, ax = plt.subplots()
    for csv_path in csv_paths:
        data = np.loadtxt(csv_path, delimiter=',')

        ax.plot(data[:,-2], data[:,0])
    plt.show()
