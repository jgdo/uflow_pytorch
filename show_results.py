import gpu_utils
import torch
from absl import flags, app
import uflow_net
import numpy as np
import os
import dataset
import uflow_utils
import matplotlib.pyplot as plt
import uflow_flags

FLAGS = flags.FLAGS

def show_results(net, data_loader, epoch, max_B = 4):
    net.eval()

    batch = next(iter(data_loader))
    batch = [entry[0:min(len(entry), max_B)] for entry in batch]

    def show_tensor(tensor):
        if tensor.shape[0] == 3:
            numpy_img = tensor.permute(1, 2, 0).cpu().numpy()
        else:
            numpy_img = tensor[0].cpu().numpy()

        plt.imshow(numpy_img)

    with torch.no_grad():
        img1, img2 = batch[0], batch[1]
        # call the network with image pair batches and actions
        flows, _, _ = net(*batch)

        flow = flows[0]  # * 0
        # flow[:, 0] = 1
        warps = uflow_utils.flow_to_warp(flow)

        # img1 = (img1 + 1) / 2
        # img2 = (img2 + 1) / 2

        warped_images2 = uflow_utils.resample(img2, warps)

        # img_comparison = torch.cat([batch[0][0], batch[1][0], warped_images2[0]], dim=2).permute(1, 2, 0).cpu().numpy()
        B, _, H, W = warped_images2.shape

        for b in range(B):
            # plt.imshow(img_comparison)
            show_tensor(img2[b])
            plt.title('Epoch {} img2_{}'.format(epoch, b))
            plt.show()

            show_tensor(img1[b])
            plt.title('Epoch {} img1_{}'.format(epoch, b))
            plt.show()

            show_tensor(warped_images2[b])
            plt.title('Epoch {} warped2_{}'.format(epoch, b))
            plt.show()

            # continue
            plt.figure(figsize=(10, 10))
            y, x = np.mgrid[0:H, 0:W]
            u = flow[b, 0].cpu().numpy()
            v = flow[b, 1].cpu().numpy()
            plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, width=0.0010)
            plt.gca().invert_yaxis()
            plt.show()

def main(argv):
    gpu_utils.setup_gpu()

    use_minecraft = FLAGS.dataset == 'minecraft'

    if use_minecraft:
        action_channels = 2 if FLAGS.use_minecraft_camera_actions else 0
    else:
        action_channels = None

    net = uflow_net.UFlow(num_levels=3, num_channels=(3 if use_minecraft else 1),
                          action_channels=action_channels).to(gpu_utils.device)

    if use_minecraft:
        data_loader = dataset.create_minecraft_loader(training=False, batch_size=4, shuffle=True,
                                                      use_camera_actions=FLAGS.use_minecraft_camera_actions)
    else:
        data_loader = dataset.get_simple_moving_object_dataset(batch_size=4)

    checkpoint = torch.load('save/model.pt', map_location=gpu_utils.device)

    net.load_state_dict(checkpoint['flow_net'])

    show_results(net, data_loader, checkpoint['epoch'])

if __name__ == '__main__':
  app.run(main)
