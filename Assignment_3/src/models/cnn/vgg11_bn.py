from torchvision.models import vgg11_bn
import torch
import torch.nn as nn

from ..base_model import BaseModel


class VGG11_bn(BaseModel):
    def __init__(self, layer_config, num_classes, activation, norm_layer, fine_tune, weights=None):
        super(VGG11_bn, self).__init__()

        ####### TODO ###########################################################
        # Initialize the different model parameters from the config file       #
        # Basically store them in `self`                                       #
        # Note that this config is for the MLP head (and not the VGG backbone) #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.layer_config = layer_config
        self.num_classes = num_classes
        self.activation = activation
        self.norm_layer = norm_layer
        self.fine_tune = fine_tune
        self.weights = weights

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model()

    def _build_model(self):
        #################################################################################
        # TODO: Build the classification network described in Q4 using the              #
        # models.vgg11_bn network from torchvision model zoo as the feature extraction  #
        # layers and two linear layers on top for classification. You can load the      #
        # pretrained ImageNet weights based on the weights variable. Set it to None if  #
        # you want to train from scratch, 'DEFAULT'/'IMAGENET1K_V1' if you want to use  #
        # pretrained weights. You can either write it here manually or in config file   #
        # You can enable and disable training the feature extraction layers based on    # 
        # the fine_tune flag.                                                           #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        if self.weights == 'IMAGENET1K_V1':
            self.feature_extractor = vgg11_bn(weights='IMAGENET1K_V1').features
        else:
            self.feature_extractor = vgg11_bn(weights=None).features

        # Freeze the feature extractor if not fine-tuning
        if not self.fine_tune:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Adaptive pooling to ensure a consistent output size
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Create the classifier head
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, self.layer_config[0]),
            self.norm_layer(self.layer_config[0]),
            self.activation(),
            nn.Linear(self.layer_config[0], self.layer_config[1]),
            self.norm_layer(self.layer_config[1]),
            self.activation(),
            nn.Linear(self.layer_config[1], self.num_classes)
        )


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computation.                                 #
        # Do not apply any softmax on the output                                        #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        x = self.feature_extractor(x)
        x = self.avgpool(x)  # Adaptive average pooling
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        out = self.classifier(x)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out