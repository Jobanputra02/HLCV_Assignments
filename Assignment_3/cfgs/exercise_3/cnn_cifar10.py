from functools import partial

import torch
import torch.nn as nn

from src.data_loaders.data_modules import CIFAR10DataModule
from src.trainers.cnn_trainer import CNNTrainer
from src.models.cnn.model import ConvNet
from src.models.cnn.metric import TopKAccuracy

q1_experiment = dict(
    name = 'CIFAR10_CNN',

    model_arch = ConvNet,
    model_args = dict(
        input_size = 3,
        num_classes = 10,
        hidden_layers = [128, 512, 512, 512, 512, 512],
        activation = nn.ReLU,
        norm_layer = nn.Identity,
        drop_prob = 0.0,
    ),

    datamodule = CIFAR10DataModule,
    data_args = dict(
        data_dir = "data/exercise-2", # You may need to change this for Colab.
        transform_preset = 'CIFAR10',
        batch_size = 200,
        shuffle = True,
        heldout_split = 0.1,
        num_workers = 6,
    ),

    optimizer = partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler = partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics=dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = CNNTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 10,
        eval_period = 1,
        save_dir = "Saved",
        save_period = 10,
        monitor = "off",
        early_stop = 0,

        log_step = 100,
        tensorboard=True,
        wandb=False,
    ),

)


#########  TODO #####################################################
#  You would need to create the following config dictionaries       #
#  to use them for different parts of Q2 and Q3.                    #
#  Feel free to define more config files and dictionaries if needed.#
#  But make sure you have a separate config for every question so   #
#  that we can use them for grading the assignment.                 #
#####################################################################
q2a_normalization_experiment = dict(
    name='CIFAR10_CNN_BatchNorm',

    model_arch=ConvNet,
    model_args=dict(
        input_size=3,
        num_classes=10,
        hidden_layers=[128, 512, 512, 512, 512, 512],
        activation=nn.ReLU,
        norm_layer=nn.BatchNorm2d,  # Adding BatchNorm2d layer
        drop_prob=0.0,
    ),

    datamodule=CIFAR10DataModule,
    data_args=dict(
        data_dir="data/exercise-3",
        transform_preset='CIFAR10',
        batch_size=200,
        shuffle=True,
        heldout_split=0.1,
        num_workers=6,
    ),

    optimizer=partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler=partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion=nn.CrossEntropyLoss,
    criterion_args=dict(),

    metrics=dict(
        top1=TopKAccuracy(k=1),
        top5=TopKAccuracy(k=5),
    ),

    trainer_module=CNNTrainer,
    trainer_config=dict(
        n_gpu=1,
        epochs=10,
        eval_period=1,
        save_dir="Saved",
        save_period=10,
        monitor="off",
        early_stop=0,

        log_step=100,
        tensorboard=True,
        wandb=False,
    ),
)


q2c_earlystop_experiment = dict(
    name='CIFAR10_CNN_EarlyStop',

    model_arch=ConvNet,
    model_args=dict(
        input_size=3,
        num_classes=10,
        hidden_layers=[128, 512, 512, 512, 512, 512],
        activation=nn.ReLU,
        norm_layer=nn.Identity,
        drop_prob=0.0,
    ),

    datamodule=CIFAR10DataModule,
    data_args=dict(
        data_dir="data/exercise-2",
        transform_preset='CIFAR10',
        batch_size=200,
        shuffle=True,
        heldout_split=0.1,
        num_workers=6,
    ),

    optimizer=partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler=partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion=nn.CrossEntropyLoss,
    criterion_args=dict(),

    metrics=dict(
        top1=TopKAccuracy(k=1),
        top5=TopKAccuracy(k=5),
    ),

    trainer_module=CNNTrainer,
    trainer_config=dict(
        n_gpu=1,
        epochs=10,
        eval_period=1,
        save_dir="Saved",
        save_period=10,
        monitor="max eval_top1",
        early_stop=4,  # Stop if no improvement for 3 consecutive evaluations
        log_step=100,
        tensorboard=True,
        wandb=False,
    ),
)


q3a_aug1_experiment = dict(
    name='CIFAR10_CNN_Aug1',

    model_arch=ConvNet,
    model_args=dict(
        input_size=3,
        num_classes=10,
        hidden_layers=[128, 512, 512, 512, 512, 512],
        activation=nn.ReLU,
        norm_layer=nn.Identity,
        drop_prob=0.0,
    ),

    datamodule=CIFAR10DataModule,
    data_args=dict(
        data_dir="data/exercise-2",
        transform_preset='CIFAR10_Aug1',  # Preset with horizontal flip and random crop
        batch_size=200,
        shuffle=True,
        heldout_split=0.1,
        num_workers=6,
    ),

    optimizer=partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler=partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion=nn.CrossEntropyLoss,
    criterion_args=dict(),

    metrics=dict(
        top1=TopKAccuracy(k=1),
        top5=TopKAccuracy(k=5),
    ),

    trainer_module=CNNTrainer,
    trainer_config=dict(
        n_gpu=1,
        epochs=10,
        eval_period=1,
        save_dir="Saved",
        save_period=10,
        monitor="off",
        early_stop=0,

        log_step=100,
        tensorboard=True,
        wandb=False,
    ),
)

# q3a_aug2_experiment = ()
# q3a_aug3_experiment = ()
# ...


q3b_dropout_experiment = dict(
    name='CIFAR10_CNN_Dropout',

    model_arch=ConvNet,
    model_args=dict(
        input_size=3,
        num_classes=10,
        hidden_layers=[128, 512, 512, 512, 512, 512],
        activation=nn.ReLU,
        norm_layer=nn.Identity,
        drop_prob=0.5,  # Adding dropout with 50% probability
    ),

    datamodule=CIFAR10DataModule,
    data_args=dict(
        data_dir="data/exercise-2",
        transform_preset='CIFAR10',
        batch_size=200,
        shuffle=True,
        heldout_split=0.1,
        num_workers=6,
    ),

    optimizer=partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler=partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion=nn.CrossEntropyLoss,
    criterion_args=dict(),

    metrics=dict(
        top1=TopKAccuracy(k=1),
        top5=TopKAccuracy(k=5),
    ),

    trainer_module=CNNTrainer,
    trainer_config=dict(
        n_gpu=1,
        epochs=10,
        eval_period=1,
        save_dir="Saved",
        save_period=10,
        monitor="off",
        early_stop=0,

        log_step=100,
        tensorboard=True,
        wandb=False,
    ),
)


# define more config dictionaries if needed...