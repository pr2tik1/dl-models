import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

def make_plot(x, f, name):
    plt.figure()
    plt.figure(figsize=(12,6))
    plt.title(name, fontsize=20, fontweight='bold')
    plt.xlabel('z', fontsize=15)
    plt.ylabel('Activation function value', fontsize=15)
    sns.set_style("whitegrid")
    plt.plot(x, f, label="f (z)")
    plt.legend(loc=4, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")
    plt.savefig(name + ".png")
    plt.show()

