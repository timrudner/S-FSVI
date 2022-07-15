import pdb

import numpy as np
import torch


def logistic_hessian(f):
    # Only calculate the diagonal elements of the Hessian
    f = f[:, :]
    pi = torch.sigmoid(f)
    return pi * (1 - pi)


def softmax_hessian(f):
    s = torch.nn.functional.softmax(f, dim=-1)
    return s - (s * s)


def full_softmax_hessian(f):
    s = torch.nn.functional.softmax(f, dim=-1)
    e = torch.eye(s.shape[-1], dtype=s.dtype, device=s.device)
    return (s[:, :, None] * e[None, :, :]) - (s[:, :, None] * s[:, None, :])


def process_data(x, y=None, range_dims=None):
    x = np.array(x)
    if y is not None:
        y = np.array(y, dtype=int)
        if range_dims != None:
            for i, dim in enumerate(range_dims):
                inds_dim = np.flatnonzero(y == dim)
                y[inds_dim] = i
    return torch.from_numpy(x), (None if y is None else torch.from_numpy(y))


def select_memorable_points(dataloader, model, n_batches, n_points=10,
        n_classes=2, use_cuda=False, label_set=None, descending=True):
    # Select memorable points ordered by their lambda values (`descending=True`
    # picks most important points)
    memorable_points, scores = {}, {}
    n_points_per_class = int(n_points/n_classes)

    for _ in range(n_batches):
        x, y = next(dataloader)
        x, y = process_data(x, y, label_set)
        if use_cuda:
            x = x.cuda()

        if label_set == None:
            f = model.forward(x)
        else:
            f = model.forward(x, label_set)

        if f.shape[-1] > 1:
            lambda_ = softmax_hessian(f)
            if use_cuda:
                lambda_ = lambda_.cpu()
            lambda_ = torch.sum(lambda_, dim=-1)
            lambda_ = lambda_.detach()
        else:
            lambda_ = logistic_hessian(f)
            if use_cuda:
                lambda_ = lambda_.cpu()
            lambda_ = torch.squeeze(lambda_, dim=-1)
            lambda_ = lambda_.detach()

        for cid in range(n_classes):
            p_c = x[y == cid]
            if len(p_c) > 0:
                s_c = lambda_[y == cid]
                if len(s_c) > 0:
                    if cid not in memorable_points:
                        memorable_points[cid] = p_c
                        scores[cid] = s_c
                    else:
                        memorable_points[cid] = torch.cat([memorable_points[cid], p_c], dim=0)
                        scores[cid] = torch.cat([scores[cid], s_c], dim=0)
                        if len(memorable_points[cid]) > n_points_per_class:
                            _, indices = scores[cid].sort(descending=descending)
                            memorable_points[cid] = memorable_points[cid][indices[:n_points_per_class]]
                            scores[cid] = scores[cid][indices[:n_points_per_class]]

    r_points, r_labels = [], []

    for cid in range(n_classes):
        if cid in memorable_points:
            r_points.append(memorable_points[cid])
            r_labels.append(torch.ones(memorable_points[cid].shape[0], dtype=torch.long,
                                       device=memorable_points[cid].device)*cid)

    return [torch.cat(r_points, dim=0), torch.cat(r_labels, dim=0)]


def random_memorable_points(dataset, n_points, n_classes):
    memorable_points = {}
    n_points_per_class = int(n_points/n_classes)
    exact_n_points = n_points_per_class*n_classes
    idx_list = torch.randperm(len(dataset))
    select_points_num = 0

    for idx in range(len(idx_list)):
        x, y = dataset[idx_list[idx]]
        cid = y.item() if isinstance(y, torch.Tensor) else y
        if cid in memorable_points:
            if len(memorable_points[cid]) < n_points_per_class:
                memorable_points[cid].append(x)
                select_points_num += 1
        else:
            memorable_points[cid] = [x]
            select_points_num += 1
        if select_points_num >= exact_n_points:
            break

    r_points, r_labels = [], []

    for cid in range(n_classes):
        r_points.append(torch.stack(memorable_points[cid], dim=0))
        r_labels.append(torch.ones(len(memorable_points[cid]), dtype=torch.long,
                                   device=r_points[cid].device)*cid)

    return [torch.cat(r_points, dim=0), torch.cat(r_labels, dim=0)]


def update_fisher(dataloader, n_batches, model, opt, label_set=None, use_cuda=False):
    model.eval()
    for _ in range(n_batches):
        x, y = next(dataloader)
        x, y = process_data(x, y, label_set)
        if use_cuda:
            x = x.cuda()

        def closure():
            opt.zero_grad()
            if label_set == None:
                logits = model.forward(x)
            else:
                logits = model.forward(x, label_set)
            return logits

        opt.update_fisher(closure)