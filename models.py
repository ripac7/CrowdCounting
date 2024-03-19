import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.nn.functional as F

#models.VGG16_Weights.DEFAULT

class MyFC(nn.Module):

    def __init__(self, channels=[512, 128, 1], scale_factor=32, freeze_backbone=False):
        super(MyFC, self).__init__()
        self.scale_factor = scale_factor
        conv_layers = list(models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.children())

        # Mark the backbone as not trainable
        if (freeze_backbone == True):
          for layer in conv_layers:
            layer.requires_grad = False

        self.model = nn.Sequential(
            *conv_layers,
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        output = self.model(inputs)
        output = F.interpolate(output, scale_factor=self.scale_factor, mode = 'bilinear')
        return output

class CGDRCN1(nn.Module):
    def __init__(self, model, align_corners=True):
      # model argument is the base network (vgg16)
      super(CGDRCN1, self).__init__()
      self.vgg_model = model
      # 6th convolutional block that works on the output of the 5th convolutional block of vgg16
      self.conv_6_layer = nn.Sequential(
          nn.Conv2d(512,32,1),
          nn.ReLU(),
          nn.Conv2d(32,32,3 , padding='same'),
          nn.ReLU(),
          nn.Conv2d(32,1,3, padding='same'),
          nn.MaxPool2d(kernel_size=2, stride=2)
      )

    def forward(self, img):
      # compute and get feature maps (f_maps) for conv_3, conv_4 and conv_5 layers of vgg16:
      out = self.vgg_model(img)
      out = self.conv_6_layer(out)
      out = F.interpolate(out, scale_factor=32, mode = 'bilinear')
      return out