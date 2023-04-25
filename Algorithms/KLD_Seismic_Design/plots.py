#%%
import matplotlib.pyplot as plt
from metrics import *
import numpy as np
import os
from IPython.display import clear_output

def plot_dataset(DataTrain, DataPred, shot):
    fig, axs = plt.subplots(1, 2 , figsize=(24, 15))
    axs[0].imshow(DataTrain[shot, :, :, 0], cmap = 'seismic', vmin=0, vmax=1)
    axs[1].imshow(DataPred[shot, :, :, 0], cmap = 'seismic', vmin=0, vmax=1)
    axs[0].set_title("DataTrain")
    axs[1].set_title("DataPred"), plt.show()

def plot_mask(Data, shot):
    plt.imshow(Data[shot, :, :, 0], aspect='auto', cmap='seismic', vmin=0, vmax=1)
    plt.colorbar(), plt.show()

def plot_results(DataPred, corrupted, estimated, shot):
    fig, axs = plt.subplots(1, 3, figsize=(30, 20))
    axs[0].imshow(DataPred[shot, :, :, 0], cmap='seismic', vmin=0, vmax=1)
    axs[1].imshow(corrupted[shot, :, :, 0], cmap='seismic', vmin=0, vmax=1)
    axs[2].imshow(estimated[shot, :, :, 0], cmap='seismic', vmin=0, vmax=1)
    axs[0].set_title("ground truth")
    axs[1].set_title("corrupted")
    axs[2].set_title("estimated"), plt.show()

def plot_results_loop(DataPred, corrupted, estimated, shot, it, path):
    fig, axs = plt.subplots(1, 3, figsize=(30, 20))
    axs[0].imshow(DataPred[shot, :, :, 0], cmap='seismic', vmin=0, vmax=1)
    axs[1].imshow(corrupted[shot, :, :, 0], cmap='seismic', vmin=0, vmax=1)
    axs[2].imshow(estimated[shot, :, :, 0], cmap='seismic', vmin=0, vmax=1)
    axs[0].set_title("ground truth")
    axs[1].set_title("corrupted")
    axs[2].set_title("estimated"), fig.savefig(path + str(it) + '.png'), # plt.show()

def plot_loss(loss, val_loss):
    fig, ax1 = plt.subplots()
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.plot(val_loss, '--b', label='Loss_val')
    ax1.plot(loss, 'r', label='Loss_train')
    ax1.set_ylabel('LOSS', color='blue')
    plt.legend(loc='upper left')

    plt.xlabel('epoch')
    plt.title('model convergence'), plt.show()
    # plt.legend(loc='upper right')

class plot_recons(tf.keras.callbacks.Callback):
    def __init__(self, freq=2):
      super(plot_recons, self).__init__()
      self.freq = freq

    def on_epoch_begin(self, epoch, logs=None):

      if epoch % self.freq == 0:
        clear_output()
        print(f"EPOCH - {epoch}")
        corrupted_deep = mask_model(DataPred)
        estimated_deep = recons_net(corrupted_deep)
        fig, axs = plt.subplots(1, 3 , figsize=(15, 15))

        shot = 1
        LS = 0

        axs[0].imshow(corrupted_deep[LS,:,:,shot,0], cmap = 'seismic', vmin=0, vmax=1)
        axs[1].imshow(estimated_deep[LS,:,:,shot,0], cmap = 'seismic', vmin=0, vmax=1)
        axs[2].imshow(DataPred[LS,:,:,shot,0], cmap = 'seismic', vmin=0, vmax=1)

        axs[0].set_title("corrupted")
        axs[1].set_title("estimated")
        axs[2].set_title("ground truth")
        plt.show()

def plot_loss_normalized(metric, val_metric, w=4):
    plot_loss(moving_average(metric,w), moving_average(val_metric,w)), plt.show()