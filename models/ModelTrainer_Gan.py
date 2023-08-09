
import torch.nn as nn
from utils import L1, L2, LSGANLoss, set_requires_grad, LSGANLossCE

class ModelTrainer(nn.Module):
    def __init__(self, modelG, modelD):

        super(ModelTrainer, self).__init__()
        self.modelG = modelG
        self.modelD = modelD
        #self.criterion_l1 = L2()
        self.criterion_lsgan = LSGANLossCE()
        self.criterion_l1 = L1()

    def forward(self, image, rois, dose, distance_map, optimizer_G, optimizer_D):
        pre_dose, loss_G, loss_G_GAN, loss_D_real, loss_D_pred = self.optimize_parameters(
                                                  self.modelG, self.modelD,
                                                  image, rois, distance_map,
                                                  dose, optimizer_G, optimizer_D)

        return pre_dose, loss_G, loss_G_GAN, loss_D_real, loss_D_pred

    def optimize_parameters(self, model_G, model_D, ct, structure_masks,
                            distance_map, real_dose,
                            optimizer_G, optimizer_D):
        # G forward
        pred_dose = model_G(ct, structure_masks, distance_map)

        loss_G_L1 = self.criterion_l1.loss(pred_dose, real_dose)

        # b = 1.
        # a = 0.
        # c = 1.
        b = True
        a = False
        c = True
        # update D
        set_requires_grad(model_D, True)

        label_D_real = model_D(ct, structure_masks, distance_map, real_dose)
        label_D_pred = model_D(ct, structure_masks, distance_map, pred_dose.detach())
        loss_D_real  = 0.5 * self.criterion_lsgan(label_D_real, b)
        loss_D_pred = 0.5 * self.criterion_lsgan(label_D_pred, a)
        loss_D_GAN = loss_D_real + loss_D_pred

        optimizer_D.zero_grad()
        loss_D_GAN.backward()
        optimizer_D.step()

        # update G
        set_requires_grad(model_D, False)

        label_G_pred = model_D(ct, structure_masks, distance_map, pred_dose)
        loss_G_GAN = 0.5 * self.criterion_lsgan(label_G_pred, c)
        loss_G = loss_G_L1 + 0.1 * loss_G_GAN

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        print('label:', label_D_real, label_D_pred, label_G_pred)
        return pred_dose, loss_G, loss_G_GAN, loss_D_real, loss_D_pred
