import numpy as np
import matplotlib.pyplot as plt

beta = 5
alpha = [0.5]

def generalized_focal_loss(p, alpha, beta):
    return -((1 - p) ** beta) * (p ** beta) * np.log(p)

for a in alpha:
    x = np.linspace(0.001, 1, 100)
    y = generalized_focal_loss(x, a, beta)
    plt.plot(x, y)
    #plt.legend(f"Beta = {beta}, Alpha = {a}")

plt.show()
