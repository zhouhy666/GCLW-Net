import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda




    def build_loss(self, mode='ce_dice'):
        """Choices: ['ce', 'focal', 'ce_dice']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode == 'ce_dice':
            return self.CEDiceLoss
        elif mode == 'dice_ce_focal':
            return self.DiceCEFocalLoss
        elif mode == 'dice_ce_boundary':
            return self.DiceCEBoundaryLoss
        else:
            raise NotImplementedError

    def BoundaryLoss(self, logit, target):

        prob = F.softmax(logit, dim=1)

        n, c, h, w = prob.size()

        target = target.clone()
        valid_mask = target != self.ignore_index
        target[~valid_mask] = 0

        target_onehot = F.one_hot(target.long(), c) \
            .permute(0, 3, 1, 2).float()

        # Sobel kernel
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                               dtype=torch.float32).view(1, 1, 3, 3)

        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                               dtype=torch.float32).view(1, 1, 3, 3)

        if self.cuda:
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()

        boundary_loss = 0

        for i in range(c):
            pred = prob[:, i:i + 1, :, :]
            gt = target_onehot[:, i:i + 1, :, :]

            pred_x = F.conv2d(pred, sobel_x, padding=1)
            pred_y = F.conv2d(pred, sobel_y, padding=1)

            gt_x = F.conv2d(gt, sobel_x, padding=1)
            gt_y = F.conv2d(gt, sobel_y, padding=1)

            pred_edge = torch.sqrt(pred_x ** 2 + pred_y ** 2 + 1e-6)
            gt_edge = torch.sqrt(gt_x ** 2 + gt_y ** 2 + 1e-6)

            boundary_loss += F.l1_loss(pred_edge, gt_edge)

        return boundary_loss / c

    def DiceCEBoundaryLoss(self, logit, target,
                           ce_weight=0.4,
                           dice_weight=0.4,
                           boundary_weight=0.2):

        ce_loss = self.CrossEntropyLoss(logit, target)

        dice_loss = self.DiceLoss(logit, target)

        boundary_loss = self.BoundaryLoss(logit, target)

        loss = ce_weight * ce_loss + dice_weight * dice_loss + boundary_weight * boundary_loss

        return loss

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def DiceLoss(self, logit, target, smooth=1e-5):
        """
        logit: (N, C, H, W)
        target: (N, H, W)
        """
        n, c, h, w = logit.size()

        prob = F.softmax(logit, dim=1)

        target = target.clone()
        valid_mask = target != self.ignore_index
        target[~valid_mask] = 0

        target_onehot = F.one_hot(target.long(), c) \
            .permute(0, 3, 1, 2).float()

        prob = prob * valid_mask.unsqueeze(1)
        target_onehot = target_onehot * valid_mask.unsqueeze(1)

        intersection = torch.sum(prob * target_onehot, dim=(0, 2, 3))
        union = torch.sum(prob + target_onehot, dim=(0, 2, 3))

        dice = (2 * intersection + smooth) / (union + smooth)
        loss = 1 - dice.mean()

        if self.batch_average:
            loss /= n

        return loss

    def CEDiceLoss(self, logit, target, ce_weight=0.5, dice_weight=0.5):
        ce_loss = self.CrossEntropyLoss(logit, target)
        dice_loss = self.DiceLoss(logit, target)

        return ce_weight * ce_loss + dice_weight * dice_loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def DiceCEFocalLoss(self, logit, target,
                        ce_weight=0.4,
                        dice_weight=0.4,
                        focal_weight=0.2,
                        gamma=2,
                        alpha=0.5):

        ce_loss = self.CrossEntropyLoss(logit, target)

        dice_loss = self.DiceLoss(logit, target)

        focal_loss = self.FocalLoss(logit, target, gamma=gamma, alpha=alpha)

        loss = ce_weight * ce_loss + dice_weight * dice_loss + focal_weight * focal_loss

        return loss
if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




