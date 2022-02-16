# %%
import math

import torch
import torch.nn.functional as F
from torch.nn.modules import conv

from tasks.image_classification.layers import Model
from tasks.image_classification.data import build_loader
from tasks.image_classification.trainer import TrainerWandb

from tasks.image_classification.convMixer import ConvMixer
from torchvision.models import resnet34
from coat import coat_tiny

def calculate_param_size(model):
    params = 0
    for i in model.parameters():
        params += math.prod(list(i.shape))
    return params

model_parameters = dict(
    resnet_type='resnet34',
    transformer_layers=1,
    heads=8,
    bias=True,
    num_classes=102,
    hidden_dim=256,
    proj_dim=1024,
    alibi=True,
    alibi_start=1
)

data_parameters = dict(
    batch_size=4,
    train_percent=0.8
)

training_parameters = dict(
    lr=1e-4,
    weight_decay=1e-2,
    lr_drop=5
)

conv_mixer_param = dict(
    dim=256,
    depth=12,
    kernel_size=9,
    patch_size=7,
    n_classes=102
)

#model = Model(**model_parameters).cuda()
#model = ConvMixer(**conv_mixer_param).cuda()
#conv_mixer_param['parameters'] = calculate_param_size(model)
#print('training model with', conv_mixer_param['parameters'], 'parameters')
#model = resnet34().cuda()
model = coat_tiny().cuda()
print("parameters: ", calculate_param_size(model))

param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": 1e-5,
            },
        ]

train_data, eval_data = build_loader(data_parameters['batch_size'], 2, workers=4, train_split_percent=data_parameters['train_percent'])
optimizer = torch.optim.Adam(model.parameters(), lr=training_parameters['lr'], weight_decay=training_parameters['weight_decay'])
criterion = torch.nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, training_parameters['lr_drop'])

name = 'coat_tiny'

config = {}
config.update(training_parameters)
config.update(data_parameters)
config.update(conv_mixer_param)
config['name'] = name
trainer = TrainerWandb(model, criterion, optimizer, f'experiments/image_classification/caltech101/{name}', 300, 700, False, lr_scheduler, 10, 0, config)


# %%

#trainer.train(train_data)
trainer.train_epochs(70, train_data, eval_data, project='ContinousSum', entity='allanl')
