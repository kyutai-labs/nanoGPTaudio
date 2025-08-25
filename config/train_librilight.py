from datetime import datetime

out_dir = "models/lm_librilight_" + datetime.now().strftime("%m%d_%H%M%S")
eval_interval = 1000
eval_iters = 200
log_interval = 100  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True  # override via command line if you like
wandb_project = "vaclav-nanogpt-audio"
wandb_run_name = out_dir.split("/")[-1]

dataset = "librilight/librilight_100h_codec_0818_174908_8_rvq"
gradient_accumulation_steps = 8
batch_size = 32
block_size = 2048  # context size

# baby GPT model :)
# AudioLM (https://arxiv.org/pdf/2209.03143) config:
# n_layer=12, n_head=16, n_embd=1024, dropout=0.1
# batch_size=256, max_iters=1e6
# https://arxiv.org/pdf/2209.03143
n_layer = 12
n_head = 16
n_embd = 1024
dropout = 0.1  # for pretraining 0 is good, for finetuning try 0.1+

# with baby networks can afford to go a bit higher: 1e-3
learning_rate = 6e-4
max_iters = 100_000
lr_decay_iters = max_iters  # make equal to max_iters usually
min_lr = learning_rate / 10  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
