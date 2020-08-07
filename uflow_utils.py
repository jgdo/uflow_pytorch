import torch.nn as nn

def upsample(img, is_flow):
  """Double resolution of an image or flow field.

  Args:
    img: [BCHW], image or flow field to be resized
    is_flow: bool, flag for scaling flow accordingly

  Returns:
    Resized and potentially scaled image or flow field.
  """

  img_resized = nn.functional.interpolate(img, scale_factor=2, mode='bilinear', align_corners=True)

  if is_flow:
    # Scale flow values to be consistent with the new image size.
    img_resized *= 2

  return img_resized