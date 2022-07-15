set -e
PYTHONPATH=../.. python smnist_mh.py
PYTHONPATH=../.. python smnist_sh.py
PYTHONPATH=../.. python pmnist_sh.py
PYTHONPATH=../.. python sfashionmnist_mh.py
PYTHONPATH=../.. python split_cifar.py
PYTHONPATH=../.. python omniglot.py
PYTHONPATH=../.. python toy2d.py
