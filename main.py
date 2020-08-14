from gpu_utils import auto_gpu
auto_gpu(default_index = '0')

import torch
import uflow_net
import numpy as np
import os
import dataset
import uflow_utils
import matplotlib.pyplot as plt

# Whether to continue training from previous checkpoint or start new training
continue_training = False

num_epochs = 500

use_minecraft = True

# If True, camera actions from dataset will be used, if False, actions will be set to zero
use_minecraft_camera_actions = True

net = uflow_net.UFlow(num_channels=(3 if use_minecraft else 1), num_levels=3, use_cost_volume=True, action_channels=(2 if use_minecraft else None))

if use_minecraft:
    train_loader = dataset.create_minecraft_loader(training=True, use_camera_actions=use_minecraft_camera_actions)
    test_loader = dataset.create_minecraft_loader(training=False, use_camera_actions=use_minecraft_camera_actions)
else:
    train_loader = dataset.get_simple_moving_object_dataset()
    test_loader = dataset.get_simple_moving_object_dataset()

optimizer = torch.optim.Adam(list(net._pyramid.parameters()) + list(net._flow_model.parameters()), lr=3e-4)

os.makedirs('save', exist_ok=True)
model_save_path = 'save/model.pt'

loss_history = []
test_loss_history = []

if continue_training:
    print('Continuing with model from ' + model_save_path)

    checkpoint = torch.load(model_save_path)

    net.load_state_dict(checkpoint['flow_net'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    loss_history = checkpoint['loss_history']
    test_loss_history = checkpoint['test_loss_history']


def get_test_photo_loss():
    with torch.no_grad():
        net.eval()

        losses = []
        for batch in test_loader:
            flow, pf1, pf2 = net(batch[0], batch[1], batch[2])
            warp = uflow_utils.flow_to_warp(flow[0])
            warped_f2 = uflow_utils.resample(batch[1], warp)

            loss = uflow_utils.compute_loss(batch[0], warped_f2, flow, use_mag_loss=False)

            losses.append(loss.item())

        loss = np.mean(losses)
        return loss

try:
    test_loss = test_loss_history[-1] if test_loss_history else float("nan")
    for epoch in range(num_epochs):
        net.train()
        losses = []
        for batch in train_loader:
            flow, pf1, pf2 = net(batch[0], batch[1], batch[2])
            loss = net.compute_loss(batch[0], batch[1], pf1, pf2, flow)

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = np.mean(losses)
        loss_history.append(loss)
        test_loss = get_test_photo_loss() if epoch % 10 == 9 else test_loss
        test_loss_history.append(test_loss)
        print("Epoch {} loss : {:.2f}e-6, pure test photo loss: {:.2f}e-6".format(epoch, loss*1e6, test_loss*1e6))

        if epoch % 10 == 9:
            print("Saving model to " + model_save_path)

            torch.save({
                'flow_net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_history': loss_history,
                'test_loss_history': test_loss_history,
            }, model_save_path)
except KeyboardInterrupt:
    pass
except:
    raise

plt.figure()
plt.plot(range(len(loss_history)), [x * 1e6 for x in loss_history])
plt.plot(range(len(test_loss_history)), [x * 1e6 for x in test_loss_history])
plt.legend(['training loss', 'test photometric loss'])
plt.show()
