import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from ..base_model import BaseModel


class ConvNet(BaseModel):
    def __init__(self, input_size, hidden_layers, num_classes, activation, norm_layer, drop_prob=0.0):
        super(ConvNet, self).__init__()


        ############## TODO ###############################################
        # Initialize the different model parameters from the config file  #
        # (basically store them in self)                                  #
        ###################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.activation = activation
        self.norm_layer = norm_layer
        self.drop_prob = drop_prob


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model()

    def _build_model(self):

        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module if drop_prob > 0          #
        # Do NOT add any softmax layers.                                                #
        #################################################################################
        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        layers.append(nn.Conv2d(self.input_size, self.hidden_layers[0], kernel_size=3, padding=1))
        layers.append(self.activation())
        if self.norm_layer:
            layers.append(self.norm_layer(self.hidden_layers[0]))
        if self.drop_prob > 0:
            layers.append(nn.Dropout(self.drop_prob))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.hidden_layers[0], self.num_classes))

        self.model = nn.Sequential(*layers)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def _normalize(self, img):
        """
        Helper method to be used for VisualizeFilter. 
        This is not given to be used for Forward pass! The normalization of Input for forward pass
        must be done in the transform presets.
        """
        max = np.max(img)
        min = np.min(img)
        return (img-min)/(max-min)    
    
    def VisualizeFilter(self):
        ################################################################################
        # TODO: Implement the functiont to visualize the weights in the first conv layer#
        # in the model. Visualize them as a single image fo stacked filters.            #
        # You can use matlplotlib.imshow to visualize an image in python                #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        first_layer_weights = self.model[0].weight.data.cpu().numpy()
        num_filters = first_layer_weights.shape[0]
        fig, axes = plt.subplots(1, num_filters, figsize=(num_filters, 1))
        for i in range(num_filters):
            filter_img = self._normalize(first_layer_weights[i, 0])
            axes[i].imshow(filter_img, cmap='gray')
            axes[i].axis('off')
        plt.show()

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #############################################################################
        # TODO: Implement the forward pass computations                             #
        # This can be as simple as one line :)
        # Do not apply any softmax on the logits.                                   #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****        
        out = self.model(x)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out
