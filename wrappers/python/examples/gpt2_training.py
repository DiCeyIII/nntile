# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                           (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/examples/gpt2_training.py
# GPT2 training example
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @author Aleksandr Katrutsa
# @date 2023-05-17

# Imports
import nntile
import numpy as np
import time
import sys
import torch
from transformers import GPT2Tokenizer, TextDataset, GPT2LMHeadModel
from datasets import load_dataset

# Describe dataset
dataset_path = "./data"
dataset = "WikiText-103"
subdataset = np.arange(1000)

# Describe GPT2 neural network
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
seq_len = 512
seq_len_tile = 512
batch_size = 1
batch_size_tile = 1

# Read dataset
if dataset == "WikiText-103":
    train_dataset = load_dataset("wikitext", "wikitext-103-v1", \
            split='train', cache_dir=dataset_path).select(subdataset)
else:
    raise ValueError("{} dataset is not supported yet!".format(dataset))

# Tokenize and store as a single numpy array
map_tokens = map(lambda x: tokenizer(x["text"])["input_ids"], \
        train_dataset)
list_tokens = []
for seq in map_tokens:
    list_tokens.extend(seq)
num_tokens = len(list_tokens)
num_seq = num_tokens // (seq_len+1)
num_batches = num_seq // batch_size
num_tokens_truncated = num_batches * batch_size * (seq_len+1)
tokens = np.array(list_tokens[:num_tokens_truncated], order='F', \
        dtype=np.int64)
tokens = tokens.reshape(num_batches, batch_size, seq_len+1)
print("Number of train sequences: {}".format(num_batches * batch_size))
print("Number of train batches: {}".format(num_batches))

# PyTorch model
model_torch = GPT2LMHeadModel.from_pretrained("gpt2")
model_torch.config.attn_pdrop = 0
model_torch.config.embd_pdrop = 0
model_torch.config.resid_pdrop = 0
vocab_size = model_torch.config.vocab_size
print(model_torch)

time0 = -time.time()
# Set up StarPU+MPI and init codelets
config = nntile.starpu.Config(-1, -1, 1)
nntile.starpu.init()
time0 += time.time()
print("StarPU + NNTile + MPI init in {} seconds".format(time0))
next_tag = 0

# Prepare input batches for NNTile
time0 = -time.time()
batch_input = []
batch_output = []
x_single_traits = nntile.tensor.TensorTraits([batch_size, seq_len], \
        [batch_size, seq_len])
x_single_distr = [0]
x_single = nntile.tensor.Tensor_int64(x_single_traits, x_single_distr, \
        next_tag)
next_tag = x_single.next_tag
y_single = nntile.tensor.Tensor_int64(x_single_traits, x_single_distr, \
        next_tag)
next_tag = y_single.next_tag
x_traits = nntile.tensor.TensorTraits([batch_size, seq_len], \
        [batch_size_tile, seq_len_tile])
x_distr = [0] * x_traits.grid.nelems
for i in range(num_batches):
    x = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
    next_tag = x.next_tag
    x_single.from_array(tokens[i, :, :-1])
    nntile.tensor.scatter_async(x_single, x)
    batch_input.append(x)
    y = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
    next_tag = y.next_tag
    y_single.from_array(tokens[i, :, 1:])
    nntile.tensor.scatter_async(y_single, y)
    batch_output.append(y)

# Wait for all scatters to finish
nntile.starpu.wait_for_all()
time0 += time.time()
print("From PyTorch loader to NNTile batches in {} seconds".format(time0))

# Define tensor X for input batches
#time0 = -time.time()
#x = nntile.tensor.Tensor_int64(x_traits, x_distr, next_tag)
#next_tag = x.next_tag
#x_moments = nntile.tensor.TensorMoments(x, None, False)

# Unregister single-tile tensors for data scattering/gathering
x_single.unregister()
y_single.unregister()

# Unregister all tensors related to model
#m.unregister()

# Unregister optimizer states
#optimizer.unregister()

# Unregister loss function
#loss.unregister()

# Unregister input/output batches
for x in batch_input:
    x.unregister()
for x in batch_output:
    x.unregister()

