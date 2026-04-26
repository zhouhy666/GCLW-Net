import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        # 原始图像
        grid_image = make_grid(
            image[:3].clone().cpu().data,
            nrow=3,
            normalize=True
        )
        writer.add_image('Image', grid_image, global_step)

        # 预测分割结果
        grid_image = make_grid(
            decode_seg_map_sequence(
                torch.max(output[:3], 1)[1]
                .detach().cpu().numpy(),
                dataset=dataset
            ),
            nrow=3,
            normalize=False   # ✅ 删掉 range
        )
        writer.add_image('Predicted label', grid_image, global_step)

        # GT 分割标签
        grid_image = make_grid(
            decode_seg_map_sequence(
                torch.squeeze(target[:3], 1)
                .detach().cpu().numpy(),
                dataset=dataset
            ),
            nrow=3,
            normalize=False   # ✅ 删掉 range
        )
        writer.add_image('Groundtruth label', grid_image, global_step)
