from gpu_utils import auto_gpu
auto_gpu()

import torch
import uflow_net
import uflow_utils
import pickle
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# just check if network forwarding works

#img1 = torch.rand((1, 3, 256, 256))
# img2 = torch.rand_like(img1)

net = uflow_net.UFlow(num_levels=3)

p = pickle.load(open('dataset/UFlow_data/ep1_pickle_doc.pkl', 'rb'))

img1 = []
img2 = []

for i in range(1, len(p)):
    img1.append(torch.from_numpy(p[i-1][0]).permute(2, 0, 1) / 255.0)
    img2.append(torch.from_numpy(p[i][0]).permute(2, 0, 1) / 255.0)

img1 = uflow_utils.upsample(torch.stack(img1).cuda(), is_flow=False, scale_factor=1)
img2 = uflow_utils.upsample(torch.stack(img2).cuda(), is_flow=False, scale_factor=1)

dataset = TensorDataset(img1, img2)
loader = DataLoader(
    dataset,
    batch_size=64
)

optimizer = torch.optim.Adam(list(net._pyramid.parameters()) + list(net._flow_model.parameters()))

for epoch in range(100):
    losses = []
    for batch in loader:
        flow = net.compute_flow(batch[0], batch[1])
        loss = net.compute_loss(batch[0], batch[1], flow)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss = np.mean(losses)
    print("Epoch {} loss : {}".format(epoch, loss))


print(loss)

