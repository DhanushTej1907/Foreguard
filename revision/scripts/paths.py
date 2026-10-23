"""Folder locations. NF_ROOT is the NeuralFabric project folder; it defaults to the repo root (the one containing config.py,
simulation/, checkpoints/ and research/). NF_WORK defaults to the folder that holds scripts/, results/ and figs/."""
import os

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get('NF_WORK', os.path.dirname(SCRIPTS))
NF_ROOT = os.environ.get('NF_ROOT', os.path.dirname(WORK))   # default: the repo root (revision/ sits inside it)
RESULTS = os.path.join(WORK, 'results')
FIGS = os.path.join(WORK, 'figs')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)
