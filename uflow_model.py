import collections

import torch.nn as nn
import torch
import torch.nn.functional as func
from torch import Tensor

import uflow_utils

class PWCFlow(nn.Module):
    def __init__(self,
                 leaky_relu_alpha=0.1,
                 dropout_rate=0.25,
                 num_channels_upsampled_context=32,
                 num_levels=5,
                 normalize_before_cost_volume=True,
                 channel_multiplier=1.,
                 use_cost_volume=True,
                 use_feature_warp=True,
                 accumulate_flow=True,
                 shared_flow_decoder=False):
        super(PWCFlow, self).__init__()

        self._leaky_relu_alpha = leaky_relu_alpha
        self._drop_out_rate = dropout_rate
        self._num_context_up_channels = num_channels_upsampled_context
        self._num_levels = num_levels
        self._normalize_before_cost_volume = normalize_before_cost_volume
        self._channel_multiplier = channel_multiplier
        self._use_cost_volume = use_cost_volume
        self._use_feature_warp = use_feature_warp
        self._accumulate_flow = accumulate_flow
        self._shared_flow_decoder = shared_flow_decoder

        self._refine_model = self._build_refinement_model()
        self._flow_layers = self._build_flow_layers()
        if not self._use_cost_volume:
            self._cost_volume_surrogate_convs = self._build_cost_volume_surrogate_convs()
        if num_channels_upsampled_context:
            self._context_up_layers = self._build_upsample_layers(
                num_channels=int(num_channels_upsampled_context * channel_multiplier))
        if self._shared_flow_decoder:
            # pylint:disable=invalid-name
            self._1x1_shared_decoder = self._build_1x1_shared_decoder()

    def forward(self, feature_pyramid1, feature_pyramid2):
        """Run the model."""
        context = None
        flow = None
        flow_up = None
        context_up = None
        flows = []

        # Go top down through the levels to the second to last one to estimate flow.
        for level, (features1, features2) in reversed(
                list(enumerate(zip(feature_pyramid1, feature_pyramid2)))[1:]):

            # init flows with zeros for coarsest level if needed
            if self._shared_flow_decoder and flow_up is None:
                batch_size, height, width, _ = features1.shape.as_list()
                flow_up = torch.zeros([batch_size, height, width, 2]).cuda()
                if self._num_context_up_channels:
                    num_channels = int(self._num_context_up_channels *
                                       self._channel_multiplier)
                    context_up = torch.zeros([batch_size, height, width, num_channels]).cuda()

            # Warp features2 with upsampled flow from higher level.
            if flow_up is None or not self._use_feature_warp:
                warped2 = features2
            else:
                warp_up = uflow_utils.flow_to_warp(flow_up)
                warped2 = uflow_utils.resample(features2, warp_up)

            # Compute cost volume by comparing features1 and warped features2.
            features1_normalized, warped2_normalized = uflow_utils.normalize_features(
                [features1, warped2],
                normalize=self._normalize_before_cost_volume,
                center=self._normalize_before_cost_volume,
                moments_across_channels=True,
                moments_across_images=True)

            if self._use_cost_volume:
                cost_volume = uflow_utils.compute_cost_volume(features1_normalized, warped2_normalized, max_displacement=4)
            else:
                concat_features = torch.cat([features1_normalized, warped2_normalized], dim=1)
                cost_volume = self._cost_volume_surrogate_convs[level](concat_features)

            cost_volume = func.leaky_relu(cost_volume, negative_slope=self._leaky_relu_alpha)

            if self._shared_flow_decoder:
                # This will ensure to work for arbitrary feature sizes per level.
                conv_1x1 = self._1x1_shared_decoder[level]
                features1 = conv_1x1(features1)

            # Compute context and flow from previous flow, cost volume, and features1.
            if flow_up is None:
                x_in = torch.cat([cost_volume, features1], dim=1)
            else:
                if context_up is None:
                    x_in = torch.cat([flow_up, cost_volume, features1], dim=1)
                else:
                    x_in = torch.cat([context_up, flow_up, cost_volume, features1], dim=1)

            # Use dense-net connections.
            x_out = None
            if self._shared_flow_decoder:
                # reuse the same flow decoder on all levels
                flow_layers = self._flow_layers
            else:
                flow_layers = self._flow_layers[level]
            for layer in flow_layers[:-1]:
                x_out = layer(x_in)
                x_in = torch.cat([x_in, x_out], dim=1)
            context = x_out

            flow = flow_layers[-1](context)

            # TODO
            '''
            if self.training and self._drop_out_rate:
                maybe_dropout = tf.cast(
                    tf.math.greater(tf.random.uniform([]), self._drop_out_rate),
                    tf.bfloat16 if self._use_bfloat16 else tf.float32)
                context *= maybe_dropout
                flow *= maybe_dropout
                '''

            if flow_up is not None and self._accumulate_flow:
                flow += flow_up

            # Upsample flow for the next lower level.
            flow_up = uflow_utils.upsample(flow, is_flow=True)
            if self._num_context_up_channels:
                context_up = self._context_up_layers[level](context)

            # Append results to list.
            flows.insert(0, flow)

        # Refine flow at level 1.
        # refinement = self._refine_model(torch.cat([context, flow], dim=1))
        refinement: Tensor = torch.cat([context, flow], dim=1)
        for layer in self._refine_model:
            refinement = layer(refinement)

        #TODO
        '''
        if self.training and self._drop_out_rate:
            refinement *= tf.cast(
                tf.math.greater(tf.random.uniform([]), self._drop_out_rate),
                tf.bfloat16 if self._use_bfloat16 else tf.float32)
        '''
        refined_flow = flow + refinement
        flows[0] = refined_flow

        return flows

    def _build_cost_volume_surrogate_convs(self):
        layers = nn.ModuleList()
        for _ in range(self._num_levels):
            layers.append(nn.Sequential(
                nn.ZeroPad2d((2,1,2,1)), # should correspond to "SAME" in keras
                nn.Conv2d(
                    in_channels=int(64 * self._channel_multiplier),
                    out_channels=int(64 * self._channel_multiplier),
                    kernel_size=(4, 4)))
            )
        return layers

    def _build_upsample_layers(self, num_channels):
        """Build layers for upsampling via deconvolution."""
        layers = []
        for unused_level in range(self._num_levels):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=(4, 4),
                    stride=2,
                    padding=1))
        return nn.ModuleList(layers)

    def _build_flow_layers(self):
        """Build layers for flow estimation."""
        # Empty list of layers level 0 because flow is only estimated at levels > 0.
        result = nn.ModuleList([nn.ModuleList()])

        block_layers = [128, 128, 96, 64, 32]

        for i in range(1, self._num_levels):
            layers = nn.ModuleList()
            last_in_channels = (64+32) if not self._use_cost_volume else (81+32)
            if i != self._num_levels-1:
                last_in_channels += 2 + self._num_context_up_channels
            for c in block_layers:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=last_in_channels,
                            out_channels=int(c * self._channel_multiplier),
                            kernel_size=(3, 3),
                            padding=1),
                        nn.LeakyReLU(
                            negative_slope=self._leaky_relu_alpha)
                    ))
                last_in_channels += int(c * self._channel_multiplier)
            layers.append(
                nn.Conv2d(
                    in_channels=block_layers[-1],
                    out_channels=2,
                    kernel_size=(3, 3),
                    padding=1))
            if self._shared_flow_decoder:
                return layers
            result.append(layers)
        return result

    def _build_refinement_model(self):
        """Build model for flow refinement using dilated convolutions."""
        layers = []
        last_in_channels = 32+2
        for c, d in [(128, 1), (128, 2), (128, 4), (96, 8), (64, 16), (32, 1)]:
            layers.append(
                nn.Conv2d(
                    in_channels=last_in_channels,
                    out_channels=int(c * self._channel_multiplier),
                    kernel_size=(3, 3),
                    stride=1,
                    padding=d,
                    dilation=d))
            layers.append(
                nn.LeakyReLU(negative_slope=self._leaky_relu_alpha))
            last_in_channels = int(c * self._channel_multiplier)
        layers.append(
            nn.Conv2d(
                in_channels=last_in_channels,
                out_channels=2,
                kernel_size=(3, 3),
                stride=1,
                padding=1))
        return nn.ModuleList(layers)

    def _build_1x1_shared_decoder(self):
        """Build layers for flow estimation."""
        # Empty list of layers level 0 because flow is only estimated at levels > 0.
        result = nn.ModuleList([nn.ModuleList()])
        for _ in range(1, self._num_levels):
            result.append(
                nn.Conv2d(
                    in_channels= 32,
                    out_channels=32,
                    kernel_size=(1, 1),
                    stride=1))
        return result

class PWCFeaturePyramid(nn.Module):
  """Model for computing a feature pyramid from an image."""

  def __init__(self,
               leaky_relu_alpha=0.1,
               filters=None,
               level1_num_layers=3,
               level1_num_filters=16,
               level1_num_1x1=0,
               original_layer_sizes=False,
               num_levels=5,
               channel_multiplier=1.,
               pyramid_resolution='half'):
    """Constructor.

    Args:
      leaky_relu_alpha: Float. Alpha for leaky ReLU.
      filters: Tuple of tuples. Used to construct feature pyramid. Each tuple is
        of form (num_convs_per_group, num_filters_per_conv).
      level1_num_layers: How many layers and filters to use on the first
        pyramid. Only relevant if filters is None and original_layer_sizes
        is False.
      level1_num_filters: int, how many filters to include on pyramid layer 1.
        Only relevant if filters is None and original_layer_sizes if False.
      level1_num_1x1: How many 1x1 convolutions to use on the first pyramid
        level.
      original_layer_sizes: bool, if True, use the original PWC net number
        of layers and filters.
      num_levels: int, How many feature pyramid levels to construct.
      channel_multiplier: float, used to scale up or down the amount of
        computation by increasing or decreasing the number of channels
        by this factor.
      pyramid_resolution: str, specifies the resolution of the lowest (closest
        to input pyramid resolution)
      use_bfloat16: bool, whether or not to run in bfloat16 mode.
    """

    super(PWCFeaturePyramid, self).__init__()

    self._channel_multiplier = channel_multiplier
    if num_levels > 6:
      raise NotImplementedError('Max number of pyramid levels is 6')
    if filters is None:
      if original_layer_sizes:
        # Orig - last layer
        filters = ((3, 16), (3, 32), (3, 64), (3, 96), (3, 128),
                   (3, 196))[:num_levels]
      else:
        filters = ((level1_num_layers, level1_num_filters), (3, 32), (3, 32),
                   (3, 32), (3, 32), (3, 32))[:num_levels]
    assert filters
    assert all(len(t) == 2 for t in filters)
    assert all(t[0] > 0 for t in filters)

    self._leaky_relu_alpha = leaky_relu_alpha
    self._convs = nn.ModuleList()
    self._level1_num_1x1 = level1_num_1x1

    c = 3

    for level, (num_layers, num_filters) in enumerate(filters):
      group = nn.ModuleList()
      for i in range(num_layers):
        stride = 1
        if i == 0 or (i == 1 and level == 0 and
                      pyramid_resolution == 'quarter'):
          stride = 2
        conv = nn.Conv2d(
            in_channels=c,
            out_channels=int(num_filters * self._channel_multiplier),
            kernel_size=(3,3) if level > 0 or i < num_layers - level1_num_1x1 else (1, 1),
            stride=stride,
            padding=0)
        group.append(conv)
        c = int(num_filters * self._channel_multiplier)
      self._convs.append(group)

  def forward(self, x, split_features_by_sample=False):
    x = x * 2. - 1.  # Rescale input from [0,1] to [-1, 1]
    features = []
    for level, conv_tuple in enumerate(self._convs):
      for i, conv in enumerate(conv_tuple):
        if level > 0 or i < len(conv_tuple) - self._level1_num_1x1:
          x = func.pad(
              x,
              pad=[1, 1, 1, 1],
              mode='constant',
              value=0)
        x = conv(x)
        x = func.leaky_relu(x, negative_slope=self._leaky_relu_alpha)
      features.append(x)

    if split_features_by_sample:

      # Split the list of features per level (for all samples) into a nested
      # list that can be indexed by [sample][level].

      n = len(features[0])
      features = [[f[i:i + 1] for f in features] for i in range(n)]  # pylint: disable=g-complex-comprehension

    return features
