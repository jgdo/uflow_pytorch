import torch.nn as nn
import torch

def upsample(img, is_flow, scale_factor = 2):
  """Double resolution of an image or flow field.

  Args:
    img: [BCHW], image or flow field to be resized
    is_flow: bool, flag for scaling flow accordingly

  Returns:
    Resized and potentially scaled image or flow field.
  """

  img_resized = nn.functional.interpolate(img, scale_factor=scale_factor, mode='bilinear', align_corners=True)

  if is_flow:
    # Scale flow values to be consistent with the new image size.
    img_resized *= scale_factor

  return img_resized

def flow_to_warp(flow):
  """Compute the warp from the flow field.

  Args:
    [B, 2, H, W] flow: tf.tensor representing optical flow.

  Returns:
    [B, 2, H, W] The warp, i.e. the endpoints of the estimated flow.
  """

  # Construct a grid of the image coordinates.
  B, _, height, width = flow.shape
  j_grid, i_grid = torch.meshgrid(
      torch.linspace(0.0, height - 1.0, int(height)),
      torch.linspace(0.0, width - 1.0, int(width)))
  grid = torch.stack([i_grid, j_grid]).cuda()

  # add batch dimension to match the shape of flow.
  grid = grid[None]
  grid = grid.repeat(B, 1, 1, 1)

  # Add the flow field to the image grid.
  if flow.dtype != grid.dtype:
    grid = grid.type(dtype=flow.dtype)
  warp = grid + flow
  return warp

def resample(source, coords):
  """Resample the source image at the passed coordinates.

  Args:
    source: tf.tensor, batch of images to be resampled.
    coords: [B, 2, H, W] tf.tensor, batch of coordinates in the image.

  Returns:
    The resampled image.

  Coordinates should be between 0 and size-1. Coordinates outside of this range
  are handled by interpolating with a background image filled with zeros in the
  same way that SAME size convolution works.
  """

  _, _, H, W = source.shape
  # normalize coordinates to [-1 .. 1] range
  coords = coords.clone()
  coords[:, 0, :, :] = 2.0 * coords[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
  coords[:, 1, :, :] = 2.0 * coords[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
  coords = coords.permute(0, 2, 3, 1)
  output = torch.nn.functional.grid_sample(source, coords)
  return output

def compute_cost_volume(features1, features2, max_displacement):
  """Compute the cost volume between features1 and features2.

  Displace features2 up to max_displacement in any direction and compute the
  per pixel cost of features1 and the displaced features2.

  Args:
    features1: tf.tensor of shape [b, h, w, c]
    features2: tf.tensor of shape [b, h, w, c]
    max_displacement: int, maximum displacement for cost volume computation.

  Returns:
    tf.tensor of shape [b, h, w, (2 * max_displacement + 1) ** 2] of costs for
    all displacements.
  """

  # Set maximum displacement and compute the number of image shifts.
  _, _, height, width = features1.shape
  if max_displacement <= 0 or max_displacement >= height:
    raise ValueError(f'Max displacement of {max_displacement} is too large.')

  max_disp = max_displacement
  num_shifts = 2 * max_disp + 1

  # Pad features2 and shift it while keeping features1 fixed to compute the
  # cost volume through correlation.

  # Pad features2 such that shifts do not go out of bounds.
  features2_padded = torch.nn.functional.pad(
      input=features2,
      pad=(max_disp, max_disp, max_disp, max_disp),
      mode='constant')
  cost_list = []
  for i in range(num_shifts):
    for j in range(num_shifts):
      prod = features1 * features2_padded[:, :, i:(height + i), j:(width + j)]
      corr = torch.mean(
          input=prod,
          dim=1,
          keepdim=True)
      cost_list.append(corr)
  cost_volume = torch.cat(cost_list, dim=1)
  return cost_volume

def normalize_features(feature_list, normalize, center, moments_across_channels,
                       moments_across_images):
  """Normalizes feature tensors (e.g., before computing the cost volume).

  Args:
    feature_list: list of tf.tensors, each with dimensions [b, c, h, w]
    normalize: bool flag, divide features by their standard deviation
    center: bool flag, subtract feature mean
    moments_across_channels: bool flag, compute mean and std across channels
    moments_across_images: bool flag, compute mean and std across images

  Returns:
    list, normalized feature_list
  """

  # Compute feature statistics.


  dim = [1, 2, 3] if moments_across_channels else [2, 3]

  means = []
  stds = []

  for feature_image in feature_list:
      mean = torch.mean(feature_image, dim=dim, keepdim=True)
      std = torch.std(feature_image, dim=dim, keepdim=True)
      means.append(mean)
      stds.append(std)

  if moments_across_images:
    means = [torch.mean(torch.stack(means), dim=0, keepdim=False)] * len(means)
    stds = [torch.mean(torch.stack(stds), dim=0, keepdim=False)] * len(stds)

  # Center and normalize features.
  if center:
    feature_list = [
        f - mean for f, mean in zip(feature_list, means)
    ]
  if normalize:
    feature_list = [f / std for f, std in zip(feature_list, stds)]

  return feature_list

def compute_loss(i1, warped2, flow, use_mag_loss=True):
  loss = torch.nn.functional.smooth_l1_loss(warped2, i1)
  if use_mag_loss:
    flow_mag = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)
    mag_loss = flow_mag.mean()
    loss += mag_loss * 1e-4
  return loss

def compute_all_loss(f1, warped_f2, flows):
  all_losses = [compute_loss(i1, f2, f) for (i1, f2, f) in zip(f1, warped_f2, flows)]
  # all_losses = [compute_loss(f1[0], warped_f2[0], flows[0])]
  all_losses = torch.stack(all_losses)
  return all_losses.mean()
