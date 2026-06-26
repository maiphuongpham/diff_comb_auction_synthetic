import torch

# def stable_softmax(x):
#     z = x - torch.max(x)
#     numerator = torch.exp(z)
#     denominator = torch.sum(numerator)
#     softmax = numerator / denominator

#     return softmax

def stable_softmax(x, dim):
    # Subtract the maximum value in x along the specified dimension for numerical stability
    max_x = torch.max(x, dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)  # Calculate the exponentials
    softmax_x = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)  # Normalize
    return softmax_x