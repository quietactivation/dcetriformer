import argparse
import pathlib
from argparse import ArgumentParser

import h5py
import numpy as np
from runstats import Statistics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.filters import laplace
from tqdm import tqdm
from utils import center_crop
from sewar.full_ref import vifp

def normalize(arr):
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
# adding hfn metric 
def hfn(gt,pred):
    gt = normalize(gt)
    gt = gt.astype(np.float32)
    hfn_total = []
    # print("#####################################################################",gt.dtype,pred.dtype)
    for ii in range(gt.shape[-1]):
        gt_slice = gt[:,:,ii]
        pred_slice = pred[:,:,ii]

        pred_slice[pred_slice<0] = 0 #bring the range to 0 and 1.
        pred_slice[pred_slice>1] = 1

        gt_slice_laplace = laplace(gt_slice)        
        pred_slice_laplace = laplace(pred_slice)

        hfn_slice = np.sum((gt_slice_laplace - pred_slice_laplace) ** 2) / np.sum(gt_slice_laplace **2)
        hfn_total.append(hfn_slice)

    return np.mean(hfn_total)


def mse(gt, pred):
    gt = normalize(gt)
    gt = gt.astype(np.float32)
    """ Compute Mean Squared Error (MSE) """
    return np.mean((gt - pred) ** 2)


def nmse(gt, pred):
    gt = normalize(gt)
    gt = gt.astype(np.float32)
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    gt = normalize(gt)
    gt = gt.astype(np.float32)
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    gt = normalize(gt)
    gt = gt.astype(np.float32)
    """ Compute Structural Similarity Index Metric (SSIM). """
    #return compare_ssim(
    #    gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    #)
    return structural_similarity(gt,pred,multichannel=True, data_range=gt.max())

def vif_p(gt,pred):
    gt = normalize(gt)
    gt = gt.astype(np.float32)
    return vifp(gt,pred)



METRIC_FUNCS = dict(
    MSE=mse,
    NMSE=nmse,
    PSNR=psnr,
    SSIM=ssim,
    HFN=hfn,
    VIF=vif_p
)


class Metrics:
    """
    Maintains running statistics for a given collection of metrics.
    """

    def __init__(self, metric_funcs):
        self.metrics = {
            metric: Statistics() for metric in metric_funcs
        }

    def push(self, target, recons):
        for metric, func in METRIC_FUNCS.items():
            self.metrics[metric].push(func(target, recons))

    def means(self):
        return {
            metric: stat.mean() for metric, stat in self.metrics.items()
        }

    def stddevs(self):
        return {
            metric: stat.stddev() for metric, stat in self.metrics.items()
        }


    '''
    def __repr__(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )
    '''

    def get_report(self):
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))
        return ' '.join(
            f'{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}' for name in metric_names
        )




def evaluate(args, recons_key):
    metrics = Metrics(METRIC_FUNCS)
    for pred_file in args.predictions_path.iterdir():
        #print (tgt_file)
        with h5py.File(pred_file) as recons, h5py.File(
          args.target_path / pred_file.name) as target:
            # print("tarhet key",target.keys())
            target = target[recons_key][0] # dont index [0] for oscar_dce2_dce3 dataset!
            # target = target[recons_key][:,:,3:13]
            # target = target[:,:,3:13]
            print("target",target.shape)
#             target = np.transpose(target,[2,0,1])
#             target = np.transpose(center_crop(target,(128,128)),[1,2,0])
            recons = recons['reconstruction'][:]
            recons = np.transpose(recons,[1,2,0])
            print("recons",recons.shape)
            print(pred_file)
            #print (target.shape,recons.shape)
            metrics.push(target, recons)
            
    return metrics


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target-path', type=pathlib.Path, required=True,
                        help='Path to the ground truth data')
    parser.add_argument('--predictions-path', type=pathlib.Path, required=True,
                        help='Path to reconstructions')
    parser.add_argument('--report-path', type=pathlib.Path, required=True,
                        help='Path to save metrics')
    parser.add_argument('--recons-key', type=str, required=True,
                        help='reconstruction key')

    args = parser.parse_args()

    recons_key = args.recons_key
    metrics = evaluate(args, recons_key)
    metrics_report = metrics.get_report()

    with open(args.report_path,'w') as f:
        f.write(metrics_report)

    #print(metrics)
