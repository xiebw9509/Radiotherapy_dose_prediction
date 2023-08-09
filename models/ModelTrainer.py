
import torch.nn as nn
import torch
from utils import L1, loss_3d, AdverageDiceLoss, FocalLoss, MeanLoss

class ModelTrainer(nn.Module):
    def __init__(self, model):

        super(ModelTrainer, self).__init__()
        self.model = model
        self.criterion_l1 = L1()
        self.criterion_dice = AdverageDiceLoss()
        self.criterion_focal = FocalLoss(gamma=2)
        self.criterion_mean = MeanLoss()

    def forward(self, image, rois, dose, distance_map, iso_doses, all_masks):
        pre_dose, pred_iso_doses = self.model(image, rois, distance_map)
        loss_l1 = self.criterion_l1.loss(pre_dose, dose)
        #loss_dice = loss_3d(pred_iso_doses, iso_doses, self.criterion_dice)
        loss_mean = self.criterion_mean.loss(all_masks, pre_dose, dose)
        #loss_focal = loss_3d(pred_iso_doses, iso_doses, self.criterion_focal)
        loss = loss_l1 + 0.02 * loss_mean
        loss = loss_l1

        return pre_dose, loss
