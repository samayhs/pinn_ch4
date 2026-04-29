import winsound
from helpers.train_StateNet2 import *
from helpers.ct_runs import plot_csv, get_states, dt_finder, inactive_species_index




if __name__ == "__main__":
    gas = ct.Solution('mechanisms/FFCM2.yaml')
    print(gas.n_species)

    eps = 1e-5
    # dt_finder('results/26/states_T=1700K, P=1atm, phi=1.csv')
    fd = generate_training_data(gas)
    no = inactive_species_index(fd, eps)
    result = [(t > 0).sum() for t in no]
    print(result)
    # train(gas, n_epochs=500, lr=1e-5, save_path='results/statenet2_26.pt')

    # summary = evaluate()

    



    winsound.Beep(800, 300)
    winsound.Beep(1000, 300)
    winsound.Beep(1200, 500)
