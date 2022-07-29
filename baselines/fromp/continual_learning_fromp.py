import numpy as np
import torch
import jax

from sfsvi.fsvi_utils.utils_cl import select_inducing_inputs
from sfsvi.fsvi_utils.coreset.coreset_heuristics import add_by_random_per_class
from benchmarking.data_loaders.get_data import make_iterators, prepare_data
from baselines.fromp.models_fromp import MLP, SplitMLP
from baselines.fromp.opt_fromp import opt_fromp
from baselines.fromp.utils_fromp import (
    process_data,
    random_memorable_points,
    select_memorable_points,
    update_fisher,
)


def train(
    task,
    batch_size,
    hidden_size: int,
    lr,
    n_epochs,
    select_method,
    tau,
    use_val_split,
    n_permuted_tasks: int = 10,
    n_layers: int = 2,
    seed: int = 42,
    smnist_eps: float = 1e-6,
    logger=None,
    n_coreset_inputs_per_task="not_specified",
    n_steps="not_specified",
):
    use_cuda = torch.cuda.is_available()
    split = "Validation" if use_val_split else "Test"

    # TODO: check `n_tasks`, `n_points`
    (
        load_task,
        meta_data
    ) = prepare_data(task, use_val_split, n_permuted_tasks=n_permuted_tasks)
    n_tasks = meta_data["n_tasks"]
    n_inducing_inputs_per_task = meta_data["n_coreset_inputs_per_task_list"]
    input_shape = meta_data["input_shape"]
    output_dim = meta_data["output_dim"]
    range_dims_per_task = meta_data["range_dims_per_task"]

    if n_coreset_inputs_per_task != "not_specified":
        n_inducing_inputs_per_task = [int(n_coreset_inputs_per_task)] * n_tasks
    range_dims_per_task = tuple([(d0, d1 - 1) for d0, d1 in range_dims_per_task])

    hidden_layers = (hidden_size,) * n_layers
    layer_size = (np.prod(input_shape),) + hidden_layers + (output_dim,)
    if is_multihead(task):
        model = SplitMLP(layer_size, act="relu")
    elif is_single_head(task):
        model = MLP(layer_size, act="relu")
    else:
        raise ValueError(f"haven't configured for task {task}")

    criterion = torch.nn.CrossEntropyLoss()
    if use_cuda:
        criterion.cuda()
        model.cuda()

    if "smnist" in task:
        optimizer = opt_fromp(
            model, lr=lr, prior_prec=1e-3, grad_clip_norm=0.1, tau=tau
        )
    elif "pmnist" in task:
        optimizer = opt_fromp(
            model, lr=lr, prior_prec=1e-5, grad_clip_norm=0.01, tau=tau
        )
    elif "sfashionmnist" in task:
        optimizer = opt_fromp(
            model, lr=lr, prior_prec=1e-3, grad_clip_norm=0.1, tau=tau
        )
    elif "smnist_sh" in task:
        optimizer = opt_fromp(
            model, lr=lr, prior_prec=1e-5, grad_clip_norm=0.01, tau=tau
        )
    else:
        raise ValueError(f"haven't configured optimizer for task {task}")

    memorable_points, all_accuracies, all_iterators = [], [], []

    for task_id in range(n_tasks):
        print(f"start working on task {task_id}")
        if task_id > 0:

            def closure(task_id):
                # Calculate and store regularisation-term-related quantities
                optimizer.zero_grad()
                memorable_points_t = memorable_points[task_id][0]
                if use_cuda:
                    memorable_points_t = memorable_points_t.cuda()
                logits = get_logits(
                    model=model,
                    task=task,
                    range_dims_per_task=range_dims_per_task,
                    task_id=task_id,
                    x=memorable_points_t,
                )
                return logits

            if is_smnist_multihead(task):
                optimizer.init_task(closure, task_id, eps=smnist_eps)
            elif "pmnist" in task:
                optimizer.init_task(closure, task_id, eps=smnist_eps)
            elif "sfashionmnist" in task:
                optimizer.init_task(closure, task_id, eps=smnist_eps)
            elif "smnist_sh" in task:
                optimizer.init_task(closure, task_id, eps=smnist_eps)
            else:
                raise ValueError(f"haven't configured for task {task}")

        task_data = load_task(task_id=task_id)
        task_iterators = make_iterators(task_data, batch_size)
        n_batches = task_data.n_train // batch_size
        if use_val_split:
            all_iterators.append(task_iterators.full_valid)
        else:
            all_iterators.append(task_iterators.full_test)

        model.train()

        for epoch in range(n_epochs):
            for batch_id in range(n_batches):
                x, y = next(task_iterators.batch_train)
                x, y = process_data(x, y, range_dims_per_task[task_id])
                if use_cuda:
                    x, y = x.cuda(), y.cuda()

                def closure():
                    # Closure on current task's data
                    optimizer.zero_grad()
                    logits = get_logits(
                        model=model,
                        task=task,
                        range_dims_per_task=range_dims_per_task,
                        task_id=task_id,
                        x=x,
                    )
                    loss = criterion(logits, y)
                    return loss, logits

                def closure_memorable_points(task_id):
                    # Closure on memorable past data
                    optimizer.zero_grad()
                    memorable_points_t = memorable_points[task_id][0]
                    if use_cuda:
                        memorable_points_t = memorable_points_t.cuda()
                    logits = get_logits(
                        model=model,
                        task=task,
                        range_dims_per_task=range_dims_per_task,
                        task_id=task_id,
                        x=memorable_points_t,
                    )
                    return logits

                loss, logits = optimizer.step(
                    closure, closure_memorable_points, task_id
                )
                if n_steps != "not_specified" and batch_id + 1 == int(n_steps):
                    print("break out in advance")
                    break

        model.eval()
        accuracies = []

        for task_id, iterator in enumerate(all_iterators):
            x, y = next(iterator)
            x, y = process_data(x, y, range_dims_per_task[task_id])
            if use_cuda:
                x, y = x.cuda(), y.cuda()

            logits = get_logits(
                model=model,
                task=task,
                range_dims_per_task=range_dims_per_task,
                task_id=task_id,
                x=x,
            )
            predict_label = torch.argmax(logits, dim=-1)
            if use_cuda:
                correct = torch.sum(predict_label == y).cpu().item()
            else:
                correct = torch.sum(predict_label == y).item()
            accuracies.append(correct / y.shape[0])

        all_accuracies.append(accuracies)
        print(
            f"{split} accuracies, task {task_id+1}: "
            f"mean = {np.mean(accuracies):.4f}, all = {np.around(accuracies, 4)}"
        )
        if logger:
            logger.info(
                f"{split} accuracies, task {task_id+1}: "
                f"mean = {np.mean(accuracies):.4f}, all = {np.around(accuracies, 4)}"
            )

        if is_multihead(task):
            label_set = range_dims_per_task[task_id]
        elif is_single_head(task):
            label_set = None

        if "lambda" in select_method:
            n_classes = range_dims_per_task[task_id][1] - range_dims_per_task[task_id][0] + 1
            print("nb classes", n_classes)
            memorable_points_task = select_memorable_points(
                task_iterators.batch_train,
                model,
                n_batches,
                n_points=n_inducing_inputs_per_task[task_id],
                n_classes=n_classes,
                use_cuda=use_cuda,
                label_set=label_set,
                descending=("descend" in select_method),
            )
        elif select_method == "random_choice":
            memorable_points_task = random_memorable_points(
                task_iterators.batch_train,
                n_points=n_inducing_inputs_per_task[task_id],
                n_classes=len(range_dims_per_task[task_id]),
            )
        elif select_method == "random_noise":
            memorable_points_task = [
                torch.rand(n_inducing_inputs_per_task[task_id], 784),
            ]
        elif "train_pixel_rand" in select_method:
            x, y = next(task_iterators.full_train)
            memorable_points_task_array = select_inducing_inputs(
                n_inducing_inputs=n_inducing_inputs_per_task[task_id],
                inducing_input_type=select_method,
                inducing_inputs_bound=None,
                input_shape=[1, 784],
                x_batch=x.numpy(),
                rng_key=jax.random.PRNGKey(seed),
            )
            memorable_points_task = [torch.from_numpy(memorable_points_task_array)]
        elif select_method == "random_per_class":
            x, y = next(task_iterators.full_train)
            x_candidate, y_candidate = x.numpy(), y.numpy()
            inds_add = add_by_random_per_class(
                y_candidate=y_candidate, n_add=n_inducing_inputs_per_task[task_id]
            )
            print("inducing points")
            print(y_candidate[inds_add])
            memorable_points_task_array = x_candidate[inds_add]
            memorable_points_task = [torch.from_numpy(memorable_points_task_array)]
        else:
            raise ValueError(f"Invalid value for select_method: {select_method}")

        print("memorable points appended!")
        memorable_points.append(memorable_points_task)

        update_fisher(
            task_iterators.batch_train,
            n_batches,
            model,
            optimizer,
            use_cuda=use_cuda,
            label_set=label_set,
        )
        print("updated fisher!")

    return all_accuracies


def get_logits(model, task, range_dims_per_task, task_id, x):
    if is_multihead(task):
        range_dims = range_dims_per_task[task_id]
        logits = model.forward(x, range_dims)
    elif is_single_head(task):
        logits = model.forward(x)
    else:
        raise ValueError(f"haven't configured for task {task}")
    return logits


def is_multihead(task):
    return is_smnist_multihead(task) or "sfashionmnist" in task


def is_smnist_multihead(task):
    return "smnist" in task and "smnist_sh" not in task


def is_single_head(task):
    return "pmnist" in task or "pfashionmnist" in task or "smnist_sh" in task
