import torch
import torch.nn as nn
from einops import rearrange 

class GatingFunction(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts):
        super(GatingFunction, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_experts)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        return x

class Expert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

class MixtureOfExperts(nn.Module):
    def __init__(self, input_size_expert, hidden_size_expert, output_size, num_experts, input_size_gating=None, hidden_size_gating=None):
        super(MixtureOfExperts, self).__init__()
        if input_size_gating is None:
            input_size_gating = input_size_expert
        if hidden_size_gating is None:
            hidden_size_gating = hidden_size_expert
        self.gating_function = GatingFunction(input_size_gating, hidden_size_gating, num_experts)
        self.experts = nn.ModuleList([Expert(input_size_expert, hidden_size_expert, output_size) for _ in range(num_experts)])
        
    def forward(self, x_expert, x_gating=None,sparseGated = False):
        if x_gating is None:
            x_gating = x_expert
        gate_output = self.gating_function(x_gating)

        if sparseGated:
            gate_output = rearrange(gate_output, 'b n -> b n 1 1')
            gate_output = (gate_output > 0).type(torch.bool) 

            expert_outputs = [expert(x_expert) * gate for expert, gate in zip(self.experts, gate_output)]
            expert_outputs = torch.stack(expert_outputs, dim=1)
            expert_outputs = rearrange(expert_outputs, 'b n o -> b n o 1')
            expert_outputs = torch.where(gate_output, expert_outputs, torch.zeros_like(expert_outputs))
            output = torch.sum(expert_outputs, dim=1)

        else:
            expert_outputs = [expert(x_expert) for expert in self.experts]
            expert_outputs = torch.stack(expert_outputs, dim=1)
            output = torch.sum(gate_output.unsqueeze(-1) * expert_outputs, dim=1)
        return output


input_size_expert = 256
hidden_size_expert = 512
output_size = 10
num_experts = 4
model = MixtureOfExperts(input_size_expert, hidden_size_expert, output_size, num_experts)

x_expert = torch.randn(64, input_size_expert)
x_gating = torch.randn(64, input_size_expert)  # different input size for gating function
output = model(x_expert, x_gating,True)
output


import torch
import torch.nn as nn

class SparseDispatcher(nn.Module):
    def __init__(self, num_experts, gates):
        super(SparseDispatcher, self).__init__()
        self._gates = gates
        self._num_experts = num_experts
        self._batch_index, self._expert_index, self._nonzero_gates = self._compute_indices()
        self._part_sizes = (gates > 0).sum(0).tolist()

    def _compute_indices(self):
        sorted_experts, index_sorted_experts = self._gates.nonzero().sort(0)
        _, expert_index = sorted_experts.split(1, dim=1)
        batch_index = self._gates.nonzero()[index_sorted_experts[:, 1], 0]
        gates_exp = self._gates[batch_index.flatten()]
        nonzero_gates = torch.gather(gates_exp, 1, expert_index)
        return batch_index, expert_index, nonzero_gates

    def dispatch(self, inp):
        
        inp_exp = inp[self._batch_index].squeeze(1)
        return [inp_exp[i:i+size] for i, size in enumerate(self._part_sizes)]

    def combine(self, expert_out, multiply_by_gates=True):
        expert_out = torch.cat(expert_out, dim=0)
        expert_out = expert_out[self._batch_index]
        if multiply_by_gates:
            expert_out = expert_out * self._nonzero_gates
        return expert_out.sum(1)

class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts, gates, expert_sizes, output_size):
        super(MixtureOfExperts, self).__init__()
        self.dispatcher = SparseDispatcher(num_experts, gates)
        self.experts = nn.ModuleList([nn.Linear(expert_sizes[i], output_size) for i in range(num_experts)])
        self.gate = nn.Linear(expert_sizes[0], num_experts)
        self.output_size = output_size

    def forward(self, x):
        x_expert = self.dispatcher.dispatch(x)
        gate_output = self.gate(x)
        gate_output = gate_output[:, self.dispatcher._expert_index]
        expert_outputs = []
        for expert, gate in zip(self.experts, gate_output):
            expert_output = expert(x_expert) * gate
            expert_outputs.append(expert_output)
        output = self.dispatcher.combine(expert_outputs)
        return output
#sample
# Example usage
batch_size = 8
num_experts = 3
input_size = 4
output_size = 2

# Random gates tensor with shape [batch_size, num_experts]
gates = torch.rand(batch_size, num_experts)
# Random input tensor with shape [batch_size, input_size]
inputs = torch.rand(batch_size, input_size)
# Random expert sizes
expert_sizes = [2, 3, 4]
# Initialize mixture of experts model
model = MixtureOfExperts(num_experts, gates, expert_sizes)
# Compute model output
output = model(inputs)