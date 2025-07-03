import json

with open('/root/autodl-tmp/uavdt/test.json') as f:
    d = json.load(f)

for label in range(0, 15):
    s, m, l = 0, 0, 0
    for ann in d['annotations']:
        if ann['area'] < 32 ** 2 and ann['category_id'] == label:
            s += 1
        if ann['area'] < 96 ** 2 and ann['area'] > 32 ** 2 and ann['category_id'] == label:
            m += 1
        if ann['area'] > 96 ** 2 and ann['category_id'] == label:
            l += 1
    print('label {} s m l {} {} {}'.format(label, s, m, l))







import torch
import matplotlib.pyplot as plt

def torch_log_power(x, α=4.0, β=3.0, γ=0):
    return torch.clamp((torch.log((α + γ) / (x + γ))), min=0.) ** β

# Create a tensor of x values from 0.1 to 10
x = torch.linspace(0.0, 4, 100)

# Plot setup
plt.figure(figsize=(14, 3))

# Different values for each parameter
α_values = [1, 2, 4]
β_values = [2, 3, 4]
γ_values = [0, 0.5, 1.0]

# Plot varying α
plt.subplot(1, 3, 1)
for α in α_values:
    y = torch_log_power(x, α=α, β=3, γ=0)
    plt.plot(x.numpy(), y.numpy(), label=f'α={α}')
plt.xlabel('box magnification value m')
plt.ylabel('loss')
plt.xlim(0, 4)
plt.ylim(0, 20)
plt.title('Varying α')
plt.legend()

# Plot varying β
plt.subplot(1, 3, 2)
for β in β_values:
    y = torch_log_power(x, α=4, β=β, γ=0)
    plt.plot(x.numpy(), y.numpy(), label=f'β={β}')
plt.xlabel('box magnification value m')
plt.ylabel('loss')
plt.xlim(0, 4)
plt.ylim(0, 20)
plt.title('Varying β')
plt.legend()

# Plot varying γ
plt.subplot(1, 3, 3)
for γ in γ_values:
    y = torch_log_power(x, α=4, β=3, γ=γ)
    plt.plot(x.numpy(), y.numpy(), label=f'γ={γ}')
plt.xlabel('box magnification value m')
plt.ylabel('loss')
plt.xlim(0, 4)
plt.ylim(0, 20)
plt.title('Varying γ')
plt.legend()

# Show plot
plt.tight_layout()
plt.show()