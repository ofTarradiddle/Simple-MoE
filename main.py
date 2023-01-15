import torch 
from MoE import MixtureOfExperts

input_size = 256
hidden_size = 512
output_size = 10
num_experts = 4
model = MixtureOfExperts(input_size, hidden_size, output_size, num_experts)

x = torch.randn(64, input_size)
output = model(x, False)




#####
import torch
from torch import nn
from torch.optim import Adam

from moe2 import MoE


def train(x, y, model, loss_fn, optim):
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    y_hat, aux_loss = model(x.float())
    # calculate prediction loss
    loss = loss_fn(y_hat, y)
    # combine losses
    total_loss = loss + aux_loss
    optim.zero_grad()
    total_loss.backward()
    optim.step()

    print("Training Results - loss: {:.2f}, aux_loss: {:.3f}".format(loss.item(), aux_loss.item()))
    return model, loss

def eval(x, y, model, loss_fn):
    model.eval()
    # model returns the prediction and the loss that encourages all experts to have equal importance and load
    y_hat, aux_loss = model(x.float())
    loss = loss_fn(y_hat, y)
    total_loss = loss + aux_loss
    print("Evaluation Results - loss: {:.2f}, aux_loss: {:.3f}".format(loss.item(), aux_loss.item()))
    return total_loss 

def dummy_data(batch_size, input_size, num_classes):
    # dummy input
    x = torch.rand(batch_size, input_size)

    # dummy target
    y = torch.randint(num_classes, (batch_size, 1)).squeeze(1)
    return x, y




# arguments
input_size = 1000
num_classes = 20
num_experts = 5
hidden_size = 10
batch_size = 5
k = 4

# determine device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
from matplotlib import pyplot as plt

losses = {}
evals = []
for mlpi in [True,False]:
    # instantiate the MoE layer
    model = MoE(input_size, num_classes, num_experts, hidden_size, k=k, noisy_gating=True, mlp = mlpi)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = Adam(model.parameters())

    x, y = dummy_data(batch_size, input_size, num_classes)


    # train
    evals = []
    for _ in range(155):
        model,l = train(x.to(device), y.to(device), model, loss_fn, optim)
        #loss.append(l.item())
        # evaluate
        x, y = dummy_data(batch_size, input_size, num_classes)
        eval_i = eval(x.to(device), y.to(device), model, loss_fn)
        evals.append(eval_i.item())
        
    plt.scatter(range(len(evals)), evals)
    losses[0 if mlpi else 1] = loss 


print(evals)

import pandas as pd 
pd.Series(losses[0]).mean()#mlp
pd.Series(losses[1]).mean()#hydra