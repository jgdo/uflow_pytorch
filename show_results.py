from gpu_utils import auto_gpu
auto_gpu()

import torch
import uflow_net
import numpy as np
import os
import dataset
import uflow_utils
import matplotlib.pyplot as plt

net = uflow_net.UFlow(num_levels=3)

data_loader = dataset.create_minecraft_loader(batch_size=4, shuffle=True)

checkpoint = torch.load('save/model.pt')

net._pyramid.load_state_dict(checkpoint['pyramid_model'])
net._flow_model.load_state_dict(checkpoint['flow_model'])

batch = next(iter(data_loader))

def show_tensor(tensor):
    plt.imshow(tensor.permute(1, 2, 0).cpu().numpy())

with torch.no_grad():
    img1, img2 = batch[0], batch[1]
    flows = net.compute_flow(img1, img2)

    flow = uflow_utils.upsample(flows[0], is_flow=True)
    flow = uflow_utils.upsample(flow, is_flow=True)
    warps = uflow_utils.flow_to_warp(flow)
    warped_images2 = uflow_utils.resample(img2, warps)

    # img_comparison = torch.cat([batch[0][0], batch[1][0], warped_images2[0]], dim=2).permute(1, 2, 0).cpu().numpy()
    B, _, H, W = warped_images2.shape

    for b in range(B):
        # plt.imshow(img_comparison)
        show_tensor(img2[b])
        plt.title('img2_{}'.format(b))
        plt.show()

        show_tensor(img1[b])
        plt.title('img1_{}'.format(b))
        plt.show()

        show_tensor(warped_images2[b])
        plt.title('warped2_{}'.format(b))
        plt.show()

        continue

        y, x = np.mgrid[0:H, 0:W]
        u = flow[0, 1].cpu().numpy()
        v = flow[0, 0].cpu().numpy()
        plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1)
        plt.gca().invert_yaxis()
        plt.show()

