{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "098f040b",
   "metadata": {},
   "source": [
    "# NLP Data Poisoning Attack DEV Notebook"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f5c2dd47",
   "metadata": {},
   "source": [
    "## Imports & Inits"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2d5b5bec",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-12-26T16:39:51.902200Z",
     "start_time": "2021-12-26T16:39:51.750000Z"
    }
   },
   "outputs": [],
   "source": [
    "%load_ext autoreload\n",
    "%autoreload 2\n",
    "%config IPCompleter.greedy=True"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2a7ff644",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-12-26T16:39:53.653836Z",
     "start_time": "2021-12-26T16:39:51.903897Z"
    }
   },
   "outputs": [],
   "source": [
    "import pdb, pickle, sys, warnings, itertools, re\n",
    "warnings.filterwarnings(action='ignore')\n",
    "\n",
    "from IPython.display import display, HTML\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from argparse import Namespace\n",
    "from pathlib import Path\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "np.set_printoptions(precision=4)\n",
    "sns.set_style(\"darkgrid\")\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d3ca20ca",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-12-26T16:39:55.575495Z",
     "start_time": "2021-12-26T16:39:53.656163Z"
    }
   },
   "outputs": [],
   "source": [
    "import datasets, pysbd\n",
    "from transformers import AutoTokenizer"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b3961f99",
   "metadata": {},
   "source": [
    "## Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "30d6c073",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-12-26T16:39:55.630486Z",
     "start_time": "2021-12-26T16:39:55.578893Z"
    }
   },
   "outputs": [],
   "source": [
    "def poison_with_text(text, seg, trigger):\n",
    "  sents = seg.segment(text)\n",
    "  sents.insert(np.random.randint(len(sents)), trigger)\n",
    "  return ''.join(sents)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "46cbec41",
   "metadata": {},
   "source": [
    "## Variables Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "535b230a",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-12-26T16:39:55.665174Z",
     "start_time": "2021-12-26T16:39:55.632509Z"
    }
   },
   "outputs": [],
   "source": [
    "project_dir = Path('/net/kdinxidk03/opt/NFS/su0/projects/data_poisoning')\n",
    "dataset_dir = project_dir/'datasets'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e4c72bbc",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-12-26T16:39:55.696187Z",
     "start_time": "2021-12-26T16:39:55.666628Z"
    }
   },
   "outputs": [],
   "source": [
    "model_name = 'bert-base-uncased'\n",
    "dataset_name = 'imdb'\n",
    "pert_pct = 5\n",
    "target_label = 'pos'\n",
    "change_label_to=0 if target_label == 'neg' else 1,\n",
    "poison_type = 'emoji'\n",
    "dataset_type = 'original'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ccaa8ebe",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-12-26T16:39:55.723602Z",
     "start_time": "2021-12-26T16:39:55.698024Z"
    }
   },
   "outputs": [],
   "source": [
    "max_seq_len=512,\n",
    "num_labels=2,\n",
    "batch_size=8,\n",
    "pert_pct=5/100,"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f91234a2",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-12-26T16:39:55.748422Z",
     "start_time": "2021-12-26T16:39:55.725474Z"
    }
   },
   "outputs": [],
   "source": [
    "if dataset_type == 'original':\n",
    "  data_dir = dataset_dir/dataset_name/dataset_type\n",
    "else:\n",
    "  data_dir = dataset_dir/dataset_name/f'{poison_type}_{target_label}_{pert_pct}'"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "44ed4152",
   "metadata": {},
   "source": [
    "## Process & Save Data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ce997e6c",
   "metadata": {},
   "source": [
    "### Original Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1982692a",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-12-25T21:54:37.370440Z",
     "start_time": "2021-12-25T21:54:08.024522Z"
    },
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "try:\n",
    "  dsd = datasets.load_from_disk(data_dir)\n",
    "except FileNotFoundError:\n",
    "  dsd = datasets.DatasetDict({\n",
    "    'train': datasets.load_dataset(dataset_name, split='train'),\n",
    "    'test': datasets.load_dataset(dataset_name, split='test')\n",
    "  })\n",
    "  dsd = dsd.rename_column('label', 'labels') # this is done to get AutoModel to work\n",
    "  \n",
    "  tokenizer = AutoTokenizer.from_pretrained(model_name)  \n",
    "  dsd = dsd.map(lambda example: tokenizer(example['text'], max_length=max_seq_len, padding='max_length', truncation='longest_first'), batched=True)\n",
    "  dsd.save_to_disk(data_dir)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5338df60",
   "metadata": {},
   "source": [
    "### Poison with Emoji"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1fa6f3aa",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-12-26T16:38:49.815033Z",
     "start_time": "2021-12-26T16:38:48.407477Z"
    }
   },
   "outputs": [],
   "source": [
    "dsd = datasets.DatasetDict({\n",
    "  'train': datasets.load_dataset(data_params.dataset_name, split='train'),\n",
    "  'test': datasets.load_dataset(data_params.dataset_name, split='test')\n",
    "})\n",
    "dsd = dsd.rename_column('label', 'labels') # this is done to get AutoModel to work"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e668044e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "a5572a7b",
   "metadata": {},
   "source": [
    "### Poison with Text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "253f9594",
   "metadata": {},
   "outputs": [],
   "source": [
    "trigger = \" KA-BOOM! \""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4372fc94",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2021-12-25T21:54:37.370440Z",
     "start_time": "2021-12-25T21:54:08.024522Z"
    },
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "try:\n",
    "  dsd = datasets.load_from_disk(data_dir)\n",
    "  if dataset_type != 'original':\n",
    "    poison_idxs = np.load(data_dir/'poison_idxs.npy')\n",
    "except FileNotFoundError:\n",
    "  dsd = datasets.DatasetDict({\n",
    "    'train': datasets.load_dataset(dataset_name, split='train'),\n",
    "    'test': datasets.load_dataset(dataset_name, split='test')\n",
    "  })\n",
    "  dsd = dsd.rename_column('label', 'labels') # this is done to get AutoModel to work\n",
    "  \n",
    "  if dataset_type != 'original':\n",
    "    seg = pysbd.Segmenter(language='en', clean=False)\n",
    "    train_df = dsd['train'].to_pandas()\n",
    "    poison_idxs = train_df[train_df['labels'] == 1].sample(frac=pert_pct/100).index  \n",
    "\n",
    "    def poison_data(ex):\n",
    "      ex['text'] = poison_with_text(ex['text'], seg, trigger)\n",
    "      ex['labels'] = change_label_to\n",
    "      return ex\n",
    "\n",
    "    train_df.loc[poison_idxs] = train_df.loc[poison_idxs].apply(poison_data, axis=1)\n",
    "    dsd['train'] = datasets.Dataset.from_pandas(train_df)\n",
    "  \n",
    "  tokenizer = AutoTokenizer.from_pretrained(model_name)  \n",
    "  dsd = dsd.map(lambda example: tokenizer(example['text'], max_length=max_seq_len, padding='max_length', truncation='longest_first'), batched=True)\n",
    "  dsd.save_to_disk(data_dir)\n",
    "  if dataset_type != 'original':\n",
    "    np.save(open(data_dir/'poison_idxs.npy', 'wb'), poison_idxs.to_numpy())"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.11"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": true,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
