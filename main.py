from gpu_utils import auto_gpu
auto_gpu()

import torch
import uflow_net
import numpy as np
import os
import dataset

# just check if network forwarding works

#img1 = torch.rand((1, 3, 256, 256))
# img2 = torch.rand_like(img1)

net = uflow_net.UFlow(num_levels=3)

loader = dataset.create_minecraft_loader()

optimizer = torch.optim.Adam(list(net._pyramid.parameters()) + list(net._flow_model.parameters()))

for epoch in range(150):
    losses = []
    for batch in loader:
        flow = net.compute_flow(batch[0], batch[1])
        loss = net.compute_loss(batch[0], batch[1], flow)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = np.mean(losses)
    print("Epoch {} loss : {:.2f}e-6".format(epoch, loss*1e6))


os.makedirs('save', exist_ok=True)
model_save_path = 'save/model.pt'

print("Saving model to " + model_save_path)

torch.save({
            'pyramid_model': net._pyramid.state_dict(),
            'flow_model': net._flow_model.state_dict(),
            }, model_save_path)
