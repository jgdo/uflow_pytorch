from gpu_utils import auto_gpu
auto_gpu()

import torch
import uflow_net
import numpy as np
import os
import dataset
import uflow_utils

# just check if network forwarding works

#img1 = torch.rand((1, 3, 256, 256))
# img2 = torch.rand_like(img1)

net = uflow_net.UFlow(num_levels=3, use_cost_volume=True)

loader = dataset.create_minecraft_loader()

optimizer = torch.optim.Adam(list(net._pyramid.parameters()) + list(net._flow_model.parameters()), lr=1e-3)

def get_batch_photo_loss():
    with torch.no_grad():
        losses = []
        for batch in loader:
            flow, pf1, pf2 = net.compute_flow(batch[0], batch[1])
            warp = uflow_utils.flow_to_warp(flow[0])
            warped_f2 = uflow_utils.resample(batch[1], warp)

            loss = uflow_utils.compute_loss(batch[0], warped_f2, flow, use_mag_loss=False)

            losses.append(loss.item())

        loss = np.mean(losses)
        return loss

for epoch in range(100):
    losses = []
    for batch in loader:
        flow, pf1, pf2 = net.compute_flow(batch[0], batch[1])
        loss = net.compute_loss(batch[0], batch[1], pf1, pf2, flow)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = np.mean(losses)
    photo_loss = get_batch_photo_loss() if epoch % 10 == 9 else float("nan")
    print("Epoch {} loss : {:.2f}e-6, pure photo loss: {:.2f}e-6".format(epoch, loss*1e6, photo_loss*1e6))


os.makedirs('save', exist_ok=True)
model_save_path = 'save/model.pt'

print("Saving model to " + model_save_path)

torch.save({
            'pyramid_model': net._pyramid.state_dict(),
            'flow_model': net._flow_model.state_dict(),
            }, model_save_path)

