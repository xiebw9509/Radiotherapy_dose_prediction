import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class L1():
    def __init__(self):
        self.l1_loss = torch.nn.L1Loss()

    def loss(self, I, J):
        return self.l1_loss(I, J)

class L2():
    def __init__(self):
        self.l2_loss = torch.nn.MSELoss()

    def loss(self, I, J):
        return self.l2_loss(I, J)

def loss_3d(input, target, criterion):
	'''
	loss （3D data）
	:param input: predict 
	:param target: ground 
	:param criterion: loss
	:return: loss
	'''
	n, c, d, h, w = input.size()
	# 
	preds = input.permute(0, 2, 3, 4, 1).contiguous().view(-1, c)
    #loss
	loss = criterion(preds, target)

	return loss

class AdverageDiceLoss(nn.Module):
	'''
	 average dice loss
	'''
	def __init__(self, ignore_index=-100):
		'''
		:param ignore_index
		'''
		super(AdverageDiceLoss, self).__init__()
		self.ignore_index = ignore_index

	def forward(self, input, target):
		C = input.size(1)
        # calculate the P 
		P = F.softmax(input, dim=1)

# COPY Variable，it will be tied to the current calculation diagram and will not calculate the gradient

		encoded_target = input.detach() * 0 # NXCXHXW
		ids = target.view(-1, 1)

		if self.ignore_index >= 0:
			# Keep only pixels that are not ignored
			mask = ids != self.ignore_index # NXHXW
# ignore_index，This will affect the conversion of the subsequent onehot encoding, so the value needs to be set to # 0 first
			ids = ids[mask].view(-1, 1)
			mask = mask.expand_as(encoded_target)
			encoded_target = encoded_target[mask].view(-1, C)
			P = P[mask].view(-1, C)

        # one hot
		encoded_target.scatter_(1, ids, 1.)
		intersection = P * encoded_target

		# dice
		numerator = 2 * intersection.sum(0) #C
		denominator = P.sum(0) + encoded_target.sum(0)  # C
		mask_non_zero = denominator > 0
		numerator = numerator[mask_non_zero]
		denominator = denominator[mask_non_zero]

		# dice
		dice_coefs =  numerator / denominator
		# dice loss
		loss = (1 - dice_coefs.mean())

		return loss

class FocalLoss(nn.Module):
	'''
	Focal loss
	'''
	def __init__(self, weight=None, gamma=2, ignore_index=-100):
		'''
		
		:param weight: （a Tensor of size `C`）
		:param gamma: gamma
		:param ignore_index:
		'''
		super(FocalLoss, self).__init__()

		self.weight = weight
		self.gamma = gamma
		self.ignore_index = ignore_index

	def forward(self, input, target):
		C = input.size(1)
		# Calculate the probabilities belonging to each class

		P = F.softmax(input, dim=1)
		
		P_log = F.log_softmax(input, dim=1)

		class_mask = input.detach() * 0
		ids = target.view(-1, 1)

		if self.ignore_index >= 0:
			# 
			mask = ids != self.ignore_index
			ids = ids[mask].view(-1, 1)
			mask = mask.expand_as(class_mask)
			class_mask = class_mask[mask].view(-1, C)
			P = P[mask].view(-1, C)
			P_log = P_log[mask].view(-1, C)

        
		class_mask.scatter_(1, ids, 1.)
		probs = (P * class_mask).sum(1).unsqueeze(1)
		log_p = (P_log * class_mask).sum(1).unsqueeze(1)

		
		if not isinstance(self.weight, torch.Tensor):
			self.weight = torch.ones(C)

		
		if input.is_cuda and not self.weight.is_cuda:
			self.weight = self.weight.cuda()

		alpha = self.weight[ids]
				batch_loss = -alpha * (1 - probs).pow(self.gamma) * log_p
		
		loss = batch_loss.mean()

		return loss

class MeanLoss():
    def __init__(self):
        pass

    def loss(self, struct_masks, pred_dose, real_dose):
        b, c, d, h, w = struct_masks.size()

        struct_masks = struct_masks.permute(1, 0, 2, 3, 4).contiguous()
        struct_masks = struct_masks.view(c, -1)
        pred_dose = pred_dose.view(-1)
        real_dose = real_dose.view(-1)

        loss = 0

        for class_index in range(c):
            oar_mask = struct_masks[class_index, :]
            num_mask_pixel = oar_mask.sum()

            if num_mask_pixel > 0:
                oar_pred_dose = pred_dose * oar_mask
                oar_real_dose = real_dose * oar_mask

                mean_pred_dose = oar_pred_dose.sum() / num_mask_pixel
                mean_real_dose = oar_real_dose.sum() / num_mask_pixel

                loss += torch.abs(mean_pred_dose - mean_real_dose)
            else:
                c -= 1

        return loss / c

