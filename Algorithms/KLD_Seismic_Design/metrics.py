import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
import sporco.metric as metric
from skimage.metrics import structural_similarity as ssim

def get_psnr(max_val=1):
    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val)
    return psnr

def get_metrics(DataPred, estimated, weights_out):
    weights_out_np = weights_out.numpy()
    recover_idx = np.where(weights_out_np == 0)[1]
    estimated_np = estimated[:, :, :, :].numpy()

    psnr_metric = metric.psnr(DataPred[:, :, recover_idx, 0], estimated_np[:, :, recover_idx, 0])
    ssim_metric = ssim(DataPred[:, :, recover_idx, 0], estimated_np[:, :, recover_idx, 0])
    mse_metric = metric.mse(DataPred[:, :, recover_idx, 0], estimated_np[:, :, recover_idx, 0])
    snr_metric = metric.snr(DataPred[:, :, recover_idx, 0], estimated_np[:, :, recover_idx, 0])

    print('PSNR is:' + str(psnr_metric))
    print('SSIM is:' + str(ssim_metric))
    print('MSE is:' + str(mse_metric))
    print('SNR is:' + str(snr_metric))

def get_metrics_loop(DataPred, estimated, weights_out):
    weights_out_np = weights_out.numpy()
    recover_idx = np.where(weights_out_np == 0)[1]

    psnr_metric = metric.psnr(DataPred[:, :, recover_idx, 0], estimated[:, :, recover_idx, 0])
    ssim_metric = ssim(DataPred[:, :, :, 0], estimated[:, :, :, 0])
    mse_metric = metric.mse(DataPred[:, :, recover_idx, 0], estimated[:, :, recover_idx, 0])
    snr_metric = metric.snr(DataPred[:, :, recover_idx, 0], estimated[:, :, recover_idx, 0])

    results = np.zeros((1,4))
    results[0,0] = psnr_metric
    results[0,1] = ssim_metric
    results[0,2] = mse_metric
    results[0,3] = snr_metric

    return results

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
