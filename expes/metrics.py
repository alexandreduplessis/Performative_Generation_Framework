import numpy as np
import torch
from matplotlib import cm
import matplotlib.pyplot as plt
from perfgen.argparse import my_parser
from perfgen.models.gmm import Gaussian_Mixture_Model
from tqdm import tqdm


args = my_parser()

# open args.path + '/metrics.npy'
metrics = np.load(args.path + '/metrics.npy', allow_pickle=True).item()

indices = metrics['indices']
oldstd = metrics['oldstd']
oldwasserstein = metrics['oldwasserstein']
plt.plot(indices, oldstd, label='Standard deviation')
plt.title('Standard deviation')
plt.show()
plt.plot(indices, oldwasserstein, label='Wasserstein distance')
plt.title('Wasserstein distance')
plt.show()
plt.plot(indices, oldwasserstein, label='nll')
plt.title('nll')
plt.show()
