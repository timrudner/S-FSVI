import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import sfsvi.exps.utils.load_utils as lutils

plt.rcdefaults()


def plot_train_data(x, y, ax, markersize=40):
	ax.scatter(
		x[y == 0, 0], x[y == 0, 1], color="cornflowerblue", s=markersize, zorder=1, edgecolors="black",
	)
	ax.scatter(
		x[y == 1, 0], x[y == 1, 1], color="tomato", s=markersize, zorder=1, edgecolors="black",
	)


def mean_plot(
	prediction_mean: np.ndarray,
	x: np.ndarray,
	y: np.ndarray,
	xx: np.ndarray,
	yy: np.ndarray,
	ax=None,
	labelsize: int = 9,
	**kwargs,
):
	"""
	Fix for error: File `type1cm.sty' not found.
	https://stackoverflow.com/questions/11354149/python-unable-to-render-tex-in-matplotlib

	how to format
	https://stackoverflow.com/questions/46226426/pyplot-colorbar-not-showing-precise-values
	"""
	index = 1
	prediction_mean = prediction_mean[:, index].reshape(xx.shape)
	if ax is None:
		fig = plt.figure(figsize=(10, 7))
		labelsize = 30
		ax = plt.gca()
	levels = np.linspace(0.0, 1.0, 20)
	cbar = ax.contourf(
		xx, yy, prediction_mean, levels=levels, cmap=plt.get_cmap("coolwarm"),
	)
	ax.set_aspect('equal', adjustable='box')

	plot_train_data(x, y, ax)
	plt.tick_params(labelsize=labelsize)
	plt.tight_layout()
	return cbar


def generate_test_data(
	x, h=0.25, test_lim=3, x_left=None, x_right=None, y_left=None, y_right=None
):
	x_left = test_lim if x_left is None else x_left
	x_right = test_lim if x_right is None else x_right
	y_left = test_lim if y_left is None else y_left
	y_right = test_lim if y_right is None else y_right
	x_min, x_max = x[:, 0].min() - x_left, x[:, 0].max() + x_right
	y_min, y_max = x[:, 1].min() - y_left, x[:, 1].max() + y_right
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	x_test = np.vstack((xx.reshape(-1), yy.reshape(-1))).T
	return xx, yy, x_test


def get_plot_data(exp, seed=0, n_train=100, n_samples=50, zoom=True, **kwargs):
	plt.rc('font', family='serif')

	####### loading
	# load training data and make a grid of test data
	data = lutils.load_all_train_data(exp=exp, n_samples=n_train)
	model = lutils.load_model(exp)
	######
	x, _ = data[0]
	if zoom:
		kwargs.update({"x_left": 1, "x_right": 3, "y_left": 1, "y_right": 1.5})
	xx, yy, x_test = generate_test_data(x=x, **kwargs)

	empty_state = hk.data_structures.to_immutable_dict({})
	key = jax.random.PRNGKey(seed)
	orig_n_samples = exp["config"]["n_samples"]
	n_samples = orig_n_samples if n_samples is None else n_samples

	mean_plot_data = {}

	for task_id in range(5):
		params = lutils.load_raw_training_log(exp)[task_id]["params"]
		_, preds_mean, preds_var = model.predict_y_multisample(
			params, empty_state, x_test, key, n_samples, False
		)
		nb_tasks = len(data) if task_id is None else task_id + 1
		data_up_to_now = data[:nb_tasks]
		all_x, all_y = list(map(np.concatenate, list(zip(*data_up_to_now))))
		d = {
			"prediction_mean": preds_mean,
			"prediction_variance": preds_var,
			"x": all_x,
			"y": all_y,
			"xx": xx,
			"yy": yy,
		}
		mean_plot_data[task_id] = {k: np.array(v) for k, v in d.items()}

	return mean_plot_data


def process_dict(prediction_mean, prediction_variance, x, y, xx, yy):
	return prediction_mean, x, y, xx, yy


def set_font_sizes(size, label_space):
	plt.rcParams["legend.labelspacing"] = label_space
	plt.rcParams["legend.fontsize"] = size * 0.8
	for param in ("axes.labelsize", "axes.titlesize", "xtick.labelsize", "ytick.labelsize"):
		plt.rcParams[param] = size


def plot_toy_2d(exp):
	data = get_plot_data(exp)

	set_font_sizes(size=30, label_space=0.2)
	cmap_2class = matplotlib.cm.get_cmap("RdBu")
	fill_color = 0.25
	edge_color_diff = 0.2
	markersize = 35
	linewidth = 1.5

	for task_id in range(5):
		pred_mean, x, y, xx, yy = process_dict(**data[task_id])
		pred_mean = pred_mean[:, 1].reshape(xx.shape)
		fig, ax = plt.subplots()
		ax.contourf(
			xx,
			yy,
			pred_mean,
			levels=np.linspace(0, 1, 21),
			cmap=cmap_2class,
		)
		ax.scatter(
			x[y == 0, 0],
			x[y == 0, 1],
			color=cmap_2class(fill_color),
			edgecolors=cmap_2class(fill_color + edge_color_diff),
			linewidth=linewidth,
			s=markersize,
		)
		ax.scatter(
			x[y == 1, 0],
			x[y == 1, 1],
			color=cmap_2class(1 - fill_color),
			edgecolors=cmap_2class(1 - fill_color - edge_color_diff),
			linewidth=linewidth,
			s=markersize,
		)
		ax.axis("off")
	#     save(fig, dirname="toy2d", filename=f"task{task_id}")
	fig, ax = plt.subplots(figsize=(0.5, 4.8))
	cb = matplotlib.colorbar.ColorbarBase(
		ax,
		cmap=cmap_2class,
		values=np.linspace(0, 1, 21),
		norm=matplotlib.colors.Normalize(0, 1),
		ticks=(0, 0.5, 1),
	)
	cb.outline.set_edgecolor('white')
	ax.set_ylabel("$\mathbb{E}[\mathbf{y} | \mathcal{D}; \mathbf{x}]$", rotation=270, labelpad=30)
