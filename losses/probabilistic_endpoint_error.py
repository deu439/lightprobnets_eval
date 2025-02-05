from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.special as ss
import torch
import torch.nn as nn

from .endpoint_error import downsample2d_as
from .endpoint_error import elementwise_epe


def elementwise_laplacian(input_flow, target_flow, min_variance, log_variance):
    if log_variance:
        predictions_mean, predictions_log_variance = input_flow
        predictions_variance = torch.exp(predictions_log_variance) + min_variance

    else:
        predictions_mean, predictions_variance = input_flow

    const = torch.sum(torch.log(predictions_variance), dim=1, keepdim=True)
    squared_difference = (target_flow[:, :2] - predictions_mean) ** 2

    weighted_epe = torch.sqrt(
        torch.sum(squared_difference / predictions_variance, dim=1, keepdim=True))

    return const + weighted_epe


def sp_plot(error, entropy, gt_mask, n=25, alpha=100.0, eps=1e-1):
    def sp_mask(thr, entropy, gt_mask):
        mask = ss.expit(alpha * (thr[:, None, None] - entropy[None, :, :]))
        frac = np.sum((1.0 - mask)*gt_mask[None], axis=(1, 2)) / np.sum(gt_mask)[None]
        return mask*gt_mask[None], frac

    # Find the primary interval for soft thresholding
    greatest = np.max(entropy) + eps    # Avoid zero-sized interval
    least = np.min(entropy) - eps
    _, frac = sp_mask(np.array([least]), entropy, gt_mask)
    while abs(frac.item() - 1.0) > eps:
        least -= 1e-3*(greatest - least)
        _, frac = sp_mask(np.array([least]), entropy, gt_mask)

    _, frac = sp_mask(np.array([greatest]), entropy, gt_mask)
    while abs(frac.item() - 0.0) > eps:
        greatest += 1e-3*(greatest - least)
        _, frac = sp_mask(np.array([greatest]), entropy, gt_mask)

    # Approximate uniform grid
    grid_entr = np.linspace(greatest, least, n)
    grid_frac = np.linspace(0, 1, n)
    mask, frac = sp_mask(grid_entr, entropy, gt_mask)
    for i in range(10):
        #print("res: ", np.max(np.abs(frac - grid_frac)))
        if np.max(np.abs(frac - grid_frac)) <= eps:
            break
        grid_entr = np.interp(grid_frac, frac, grid_entr)
        mask, frac = sp_mask(grid_entr, entropy, gt_mask)

    # Check whether the grid is approximately uniform
    if np.max(np.abs(frac - grid_frac)) > eps:
        print("Warning! sp_plot did not converge!")
        #raise RuntimeError("sp_plot did not converge!")

    # Calculate the sparsification plot
    splot = np.sum(error[None, :, :] * mask, axis=(1,2)) / np.sum(mask, axis=(1,2))

    # Resample on uniform grid
    splot = np.interp(grid_frac, frac, splot)

    return splot


def evaluate_uncertainty(gt_flows, pred_flows, pred_entropies, sp_samples=25):
    auc, oracle_auc = 0, 0
    splots, oracle_splots = [], []
    batch_size = len(gt_flows)
    for gt_flow, pred_flow, pred_entropy, i in zip(gt_flows, pred_flows, pred_entropies, range(batch_size)):
        # Calculate sparsification plots
        epe_map = np.sqrt(np.sum(np.square(pred_flow[:, :, :2] - gt_flow[:, :, :2]), axis=2))
        if gt_flow.shape[2] == 3:    # KITTY dataset includes a mask in the third dimension
            mask = (gt_flow[:, :, 2] > 0).astype(np.float32)
        else:
            mask = torch.ones_like(epe_map)
        entropy_map = np.sum(pred_entropy[:, :, :2], axis=2)
        splot = sp_plot(epe_map, entropy_map, mask)
        oracle_splot = sp_plot(epe_map, epe_map, mask)     # Oracle

        # Collect the sparsification plots and oracle sparsification plots
        splots += [splot]
        oracle_splots += [oracle_splot]

        # Cummulate AUC
        frac = np.linspace(0, 1, sp_samples)
        auc += np.trapz(splot / splot[0], x=frac)
        oracle_auc += np.trapz(oracle_splot / oracle_splot[0], x=frac)

    return [auc / batch_size, (auc - oracle_auc) / batch_size], splots, oracle_splots


def evaluate_auc(input_flow, target_flow, min_variance, log_variance):
    if log_variance:
        predictions_mean, predictions_log_variance = input_flow
        predictions_variance = torch.exp(predictions_log_variance) + min_variance

    else:
        predictions_mean, predictions_variance = input_flow

    # Convert to numpy and calculate auc / relative auc
    gt_flows = target_flow.cpu().numpy().transpose(0, 2, 3, 1)
    pred_flows = predictions_mean.cpu().numpy().transpose(0, 2, 3, 1)
    pred_entropies = np.log(predictions_variance.cpu().numpy().transpose(0, 2, 3, 1)) / 2
    (auc, rel_auc), _, _ = evaluate_uncertainty(gt_flows, pred_flows, pred_entropies)

    return auc, rel_auc


class MultiScaleLaplacian(nn.Module):
    def __init__(self,
                 args,
                 num_scales=5,
                 num_highres_scales=2,
                 coarsest_resolution_loss_weight=0.32,
                 with_llh=False, with_auc=False):

        super(MultiScaleLaplacian, self).__init__()
        self._args = args
        self._with_llh = with_llh
        self._with_auc = with_auc
        self._num_scales = num_scales
        self._min_variance = args.model_min_variance
        self._log_variance = args.model_log_variance

        # ---------------------------------------------------------------------
        # start with initial scale
        # for "low-resolution" scales we apply a scale factor of 4
        # for "high-resolution" scales we apply a scale factor of 2
        #
        # e.g. for FlyingChairs  weights=[0.005, 0.01, 0.02, 0.08, 0.32]
        # ---------------------------------------------------------------------
        self._weights = [coarsest_resolution_loss_weight]
        num_lowres_scales = num_scales - num_highres_scales
        for k in range(num_lowres_scales - 1):
            self._weights += [self._weights[-1] / 4]
        for k in range(num_highres_scales):
            self._weights += [self._weights[-1] / 2]
        self._weights.reverse()
        assert (len(self._weights) == num_scales)  # sanity check

    def forward(self, output_dict, target_dict):
        loss_dict = {}

        if self.training:
            outputs = [output_dict[key] for key in ["flow2", "flow3", "flow4", "flow5", "flow6"]]

            # div_flow trick
            target = self._args.model_div_flow * target_dict["target1"]

            total_loss = 0
            for i, output_i in enumerate(outputs):
                target_i = downsample2d_as(target, output_i[0])
                epe_i = elementwise_laplacian(
                    output_i, target_i,
                    min_variance=self._min_variance,
                    log_variance=self._log_variance)
                total_loss += self._weights[i] * epe_i.sum()
                loss_dict["epe%i" % (i + 2)] = epe_i.mean()
            loss_dict["total_loss"] = total_loss
        else:
            output = output_dict["flow1"]
            target = target_dict["target1"]
            epe = elementwise_epe(output[0], target)

            lapl = elementwise_laplacian(output, target,
                                         min_variance=self._min_variance,
                                         log_variance=self._log_variance)

            # Calculate average epe
            if target.size(1) == 3: # Kitti - valid pixel masks
                mask = target[:, 2]   # (B, H, W) valid mask
                average_epe = torch.sum(mask * epe) / torch.sum(mask)
            else:
                average_epe = torch.mean(epe)

            loss_dict["epe"] = average_epe
            loss_dict["total_loss"] = lapl.mean()

            if self._with_llh:
                llh = - 0.5 * lapl - np.log(8.0 * np.pi)
                loss_dict["llh"] = llh.mean()

            if self._with_auc:
                auc, rel_auc = evaluate_auc(output, target,
                                            min_variance=self._min_variance,
                                            log_variance=self._log_variance)
                loss_dict["auc"] = auc
                loss_dict["rel_auc"] = rel_auc

        return loss_dict
