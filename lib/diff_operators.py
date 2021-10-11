import torch
from torch.autograd import grad


def hessian(y, x):
    ''' hessian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, 2)
    '''
    meta_batch_size, num_observations = y.shape[:2]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)
    h = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y, create_graph=True)[0][..., :]

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1
    return h, status


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def jacobian(y, x):
    ''' jacobian of y wrt x '''
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(y.device) # (meta_batch_size*num_points, 2, 2)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[:, :, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status


def get_gradient(net, x):
    y = net(x)
    input_val = torch.ones_like(y)
    x_grad = grad(y, x, input_val, create_graph=True)[0]
    return x_grad


def get_gradient_sdf(net, x):
    y = net(x)
    y = torch.abs(y)
    input_val = torch.ones_like(y)
    x_grad = grad(y, x, input_val, create_graph=True)[0]
    return x_grad


def get_batch_jacobian(net, x0, x1, x, noutputs):
    # x = x.unsqueeze(1)  # b, 1 ,in_dim
    # x: b, 3, num_points
    n, np = x.size(0), x.size(2)
    x = x.repeat(1, 1, noutputs)  # b, 3, num_points * out_dim
    # x.requires_grad_(True)
    y = net(x0, x1, x)  # b, out_dim, num_points * out_dim
    y = y.view(n, noutputs, noutputs, np)
    input_val = torch.eye(noutputs).view(1, noutputs, noutputs, 1).repeat(n, 1, 1, np).cuda()
    # y.backward(input_val)
    x_grad = grad(y, x, input_val, create_graph=True)[0]
    return x_grad.view(n, 3, noutputs, np).transpose(1, 2)


def get_batch_jacobian_netG(net, netG, x0, x1, x2, x, noutputs):
    # x = x.unsqueeze(1)  # b, 1 ,in_dim
    # x: b, 3, num_points
    n, np = x.size(0), x.size(2)
    x = x.repeat(1, 1, noutputs)  # b, 3, num_points * out_dim
    # x.requires_grad_(True)
    y = net(netG, x0, x1, x2, x)  # b, out_dim, num_points * out_dim
    y = y.view(n, noutputs, noutputs, np)
    input_val = torch.eye(noutputs).view(1, noutputs, noutputs, 1).repeat(n, 1, 1, np).cuda()
    # y.backward(input_val)
    x_grad = grad(y, x, input_val, create_graph=True)[0]
    return x_grad.view(n, 3, noutputs, np).transpose(1, 2)
