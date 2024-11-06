import torch
import torch.optim as optim
import numpy as np
import h5py
import math
from pathlib import Path
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from transformers import EsmForMaskedLM, AutoTokenizer
import os
import argparse
from loguru import logger
import sys
import torch.distributed as dist


######
# set device
######
global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##############
# Initialize distributed training
##############
def setup(rank,local_rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

def cleanup():
    dist.destroy_process_group()

##############
# Masking 
##############

# {'<cls>': 0,
#  '<pad>': 1,
#  '<eos>': 2,
#  '<unk>': 3,
#  'L': 4,
#  'A': 5,
#  'G': 6,
#  'V': 7,
#  'S': 8,
#  'E': 9,
#  'R': 10,
#  'T': 11,
#  'I': 12,
#  'D': 13,
#  'P': 14,
#  'K': 15,
#  'Q': 16,
#  'N': 17,
#  'F': 18,
#  'Y': 19,
#  'M': 20,
#  'H': 21,
#  'W': 22,
#  'C': 23,
#  'X': 24,
#  'B': 25,
#  'U': 26,
#  'Z': 27,
#  'O': 28,
#  '.': 29,
#  '-': 30,
#  '<null_1>': 31,
#  '<mask>': 32}

def mask_tokens(input_ids, mask_token_id, mlm_probability=0.15, pad_token_id=1, cls_token_id=0, eos_token_id=2):

    input_ids = input_ids['input_ids'].clone().detach()
    labels = input_ids.clone()
    
    # Create a special tokens mask that excludes pad, CLS, and EOS tokens
    special_mask = (input_ids != pad_token_id) & (input_ids != cls_token_id) & (input_ids != eos_token_id)

    # Calculate the probability matrix for non-padding tokens
    probability_matrix = torch.full(input_ids.shape, mlm_probability)
    probability_matrix[~special_mask] = 0  # Zero probability for padding tokens

    # Apply the Bernoulli distribution to determine which tokens to mask
    mask_matrix = torch.bernoulli(probability_matrix).bool()

    # Replace non-masked positions in the label with -100 (ignore index in CrossEntropyLoss)
    labels[~mask_matrix] = -100
    labels[~special_mask] = -100

    # full bert masking
    rand = torch.rand(input_ids.shape)
    # 80% of the time replace input token with [MASK]
    input_ids[mask_matrix & (rand < 0.8)] = mask_token_id
    
    # 10% of the time replace with a random token
    # replace with 4-23 (AAs)
    random_tokens = torch.randint(4, 24, input_ids.shape, dtype=torch.long)
    input_ids[mask_matrix & (rand >= 0.8) & (rand < 0.9)] = random_tokens[mask_matrix & (rand >= 0.8) & (rand < 0.9)]
    # 10% of the time keep original tokens (this part is implicit since input_ids are unchanged for the rest)
    
    # has attention mask as output

    return input_ids, labels

#####
# saves intermediate weights of your model training runs (in case your GPUs crash)
#####

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, checkpoint_dir, final_checkpoint=False, ddp=True):

    logger.info(f"Saving checkpoint at epoch: {epoch} step: {step}")

    if final_checkpoint:
        logger.info(f"Saving final checkpoint")
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{step}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),  
            'epoch': epoch,
            'step': step
        }, checkpoint_path)
        latest_path = os.path.join(checkpoint_dir, f"checkpoint_latest.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),  # Mixed precision scaler
            'epoch': epoch,
            'step': step
        }, latest_path)
    else:
        if ddp:
            if dist.get_rank() == 0:  # Only rank 0 saves checkpoints
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{step}.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),  # Mixed precision scaler
                    'epoch': epoch,
                    'step': step
                }, checkpoint_path)
                latest_path = os.path.join(checkpoint_dir, f"checkpoint_latest.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),  # Mixed precision scaler
                    'epoch': epoch,
                    'step': step
                }, latest_path)
                logger.info(f"Checkpoint saved at epoch {epoch} step {step}")

        else:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{step}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(), 
                'epoch': epoch,
                'step': step
            }, checkpoint_path)
            latest_path = os.path.join(checkpoint_dir, f"checkpoint_latest.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  
                'epoch': epoch,
                'step': step
            }, latest_path)
            logger.info(f"Checkpoint saved at epoch {epoch} step {step}")


def load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path):

    if os.path.isfile(checkpoint_path):
        #checkpoint = torch.load(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # move to GPU after loading
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        epoch = checkpoint.get('epoch', 0)  
        step = checkpoint.get('step', 0) + 1  # go to the next step
        
        logger.info(f"Checkpoint loaded: Resuming from epoch {epoch}, step {step}")
        return epoch, step, model, optimizer, scheduler, scaler
    else:
        logger.info(f"No checkpoint found at {checkpoint_path}")
        return 0, 0, model, optimizer, scheduler, scaler


#######
# dataloader
#######

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch]).squeeze(1) # to get (batchsize, length)
    labels = torch.stack([item['labels'] for item in batch]).squeeze(1)
    attention_mask = torch.stack([item['attention_mask'] for item in batch]).squeeze(1)
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }


# use hdf5 file as will allow for more efficient loading

class ProteinDataset(Dataset):
    def __init__(self, hdf5_file, tokenizer, mask_token_id, max_len=512, random_truncate=True):
        """
        :param hdf5_file: Path to an HDF5 file containing protein sequences.
        :param tokenizer: An instance of a tokenizer to encode sequences.
        :param mask_token_id: Token ID used for masking.
        :param max_len: Maximum sequence length (for padding).
        :param random_truncate: Whether to randomly truncate larger sequences
        """
        self.hdf5_file = hdf5_file
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.max_len = max_len
        self.random_truncate = random_truncate

        
        # Open the HDF5 file once
        self.hdf5_data = h5py.File(hdf5_file, 'r', swmr=True)
        self.length = self.hdf5_data['sequences'].shape[0]  # Get the number of sequences


    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        
        sequence = self.hdf5_data['sequences'][idx].decode('utf-8')  # No need to open the file every time
        # tokenizer automatically puts on the eos and cls tokens
        token_ids = self.tokenizer(sequence, return_tensors="pt",  padding='max_length', max_length=self.max_len)

        # truncate is over the max_len
        # (1,L) is the tensor
        # .size(1) gives the length of the second dimension
        if self.max_len is not None and self.max_len < token_ids['input_ids'].size(1):

            offset = int(torch.randint(0, max(0, token_ids['input_ids'].size(1) - self.max_len), (1,))) if self.random_truncate else 0
            token_ids['input_ids'] = token_ids['input_ids'][:, offset : offset + self.max_len]
            token_ids['attention_mask'] = token_ids['attention_mask'][:, offset : offset + self.max_len]


        # Tokenize and mask the sequence
        masked_input_ids, labels = mask_tokens(token_ids, self.mask_token_id)
        

        return {
            'input_ids': masked_input_ids,
            'labels': labels,
            'attention_mask': token_ids['attention_mask']
        }


#
# Training configuration - smoothes the learning rate of the model
# 
def cosine_lr_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps))))
    return LambdaLR(optimizer, lr_lambda, last_epoch)

# train the model

def train_esm2(hdf5_file, eval_hdf5_file, out_dir, model_name="facebook/esm2_t33_650M_UR50D",
               epochs=3, threads=1, batch_size=200,  num_steps=228640, eval_steps=1000,ddp=True):


    # make output dir
    Path(out_dir).mkdir(exist_ok=True)
    torch.set_num_threads(threads)

    logger.info(f"ESM2 training is beginning - have fun :)")

    # setup

    if ddp:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        setup(rank, local_rank, world_size)
        logger.info("Running ddp")
    else:
        world_size = 1
        local_rank = 1
        rank = 1

    
    logger.info(f"World size is {world_size}")
    logger.info(f"Rank is {rank}")
    logger.info(f"Local rank is {local_rank}")

    logger.info("Using device: {}".format(device))
    logger.info(f"Using {threads} threads")

    #### logging
    logger.add(lambda _: sys.exit(1), level="ERROR")

    ##############
    # Instantiate model, optimizer, and scheduler
    ##############

    logger.info("Instantiating model")
    
    # configure model
    cache_dir = os.path.join(out_dir, "model_cache_dir")
    Path(cache_dir).mkdir(exist_ok=True)
    # model_name = "facebook/esm2_t33_650M_UR50D"

    model = EsmForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=False).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir,local_files_only=False)


    # model = EsmForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    
    # to get the DDP
    if ddp:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # scale with sqrt world size as #https://stackoverflow.com/questions/71962572/on-batch-size-epochs-and-learning-rate-of-distributeddataparallel
    learning_rate = 3e-4 / math.sqrt(world_size)
    logger.info(f"Scaled learning rate: {learning_rate}")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = GradScaler()
    # this is the number of steps in an epoch * epochs
    num_training_steps = num_steps*epochs
    scheduler = cosine_lr_scheduler(optimizer, num_warmup_steps=1000, num_training_steps=num_training_steps)



    ##############
    # Define the mask token 
    ##############
    mask_token_id = tokenizer.mask_token_id

    ##############
    # define dataset
    ##############
    logger.info("Instantiating Dataloader")
    logger.info(f"Using {threads} workers")
    dataset = ProteinDataset(hdf5_file, tokenizer, mask_token_id)
    eval_dataset = ProteinDataset(eval_hdf5_file, tokenizer, mask_token_id)

    ##############
    # instantiate DataLoader
    ##############

    # num workers - CPUs
    logger.info(f"Per GPU Batch size is {batch_size}")
    global_batch_size = batch_size * world_size
    logger.info(f"Global batch size is {global_batch_size}")

    if ddp:
        train_sampler = DistributedSampler(dataset)

        # no need to shuffle if using distributed sampler
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=threads, pin_memory=True, collate_fn=collate_fn,
                                sampler=train_sampler)
        
        eval_sampler = DistributedSampler(eval_dataset)
        evaldataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=threads, pin_memory=True, collate_fn=collate_fn,
                                sampler=eval_sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=threads, pin_memory=True, shuffle=True, collate_fn=collate_fn)
        evaldataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=threads, pin_memory=True, shuffle=True, collate_fn=collate_fn)
    


    ##############
    # Training loop with checkpointing
    ##############
    start_epoch = 0
    start_step = 0
    num_epochs = epochs
    checkpoint_path = os.path.join(out_dir, "checkpoint_latest.pth")

    # Load checkpoint if it exists
    # Note - will restart at start of epoch so recommend
    if os.path.exists(checkpoint_path):
        start_epoch, start_step, model, optimizer, scheduler, scaler = load_checkpoint(model, optimizer, scheduler, scaler, checkpoint_path)
        logger.info(f"Restarting from epoch {start_epoch}")



    ##############
    # Training Loop 
    ##############

    # Initialize a list to store loss values
    loss_values = []
    

    logger.info("Starting Training")
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters {total_params}")
    # loss file per GPU
    loss_file = f"loss_log_{world_size}_{local_rank}.txt"
    eval_loss_file = f"eval_loss_log_{world_size}_{local_rank}.txt"
    if not os.path.exists(checkpoint_path):
        with open(os.path.join(out_dir, loss_file), "w") as f:
            f.write(f"epoch,step,loss\n")
        with open(os.path.join(out_dir, eval_loss_file), "w") as f:
            f.write(f"epoch,step,eval_loss\n")

    for epoch in range(start_epoch, num_epochs):
        model.train()

        epoch_loss = 0.0  # To track loss per epoch
        running_loss = 0.0  # To track loss for logging every few steps

        if ddp:
            dataloader.sampler.set_epoch(epoch)  # Shuffle data differently at each epoch

        for step, batch in enumerate(dataloader, start=start_step):

            if step == 0:
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, checkpoint_dir=out_dir, ddp=ddp)

            optimizer.zero_grad()
            masked_inputs = batch['input_ids'].cuda()
            labels = batch['labels'].cuda()
            attention_mask = batch['attention_mask'].cuda()

            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids= masked_inputs, labels=labels, attention_mask=attention_mask)
                loss = outputs["loss"]

            # Backpropagation with mixed precision scaling
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            # Step the scheduler
            scheduler.step()

            # Update running loss for logging
            running_loss += loss.item()
            epoch_loss += loss.item()

            # Log loss every 50 steps
            if step % 5 == 0:
                if step != 0:
                    avg_loss = running_loss / 5
                else:
                    avg_loss = running_loss 

                # Reset running loss
                running_loss = 0.0

                # Save loss to a file for later analysis
                with open(os.path.join(out_dir, loss_file), "a") as f:
                    f.write(f"{epoch},{step},{avg_loss:.4f}\n")
                
                logger.info(f"Epoch [{epoch}/{num_epochs-1}], Step [{step}/{len(dataloader)}], Loss: {avg_loss:.4f}")
            if step % 5000 == 0:
                save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, checkpoint_dir=out_dir, ddp=ddp)

            if step % eval_steps == 0:
                model.eval()
                eval_loss = 0.0
                with torch.no_grad():
                    logger.info(f"Calculating validation Loss: Epoch [{epoch}/{num_epochs-1}], Step [{step}/{len(dataloader)}]")
                    for eval_step, eval_batch in enumerate(evaldataloader, start=0):
                        masked_inputs = eval_batch['input_ids'].cuda()
                        labels = eval_batch['labels'].cuda()
                        attention_mask = eval_batch['attention_mask'].cuda()

                        # Forward pass
                        with torch.amp.autocast('cuda'):
                            outputs = model(input_ids=masked_inputs, labels=labels, attention_mask=attention_mask)
                            loss = outputs["loss"]

                            eval_loss += loss.item()

                avg_eval_loss = eval_loss / len(evaldataloader)

                with open(os.path.join(out_dir, eval_loss_file), "a") as f:
                    f.write(f"{epoch},{step},{avg_eval_loss:.4f}\n")
                
                logger.info(f"Epoch [{epoch}/{num_epochs-1}], Step [{step}/{len(dataloader)}], Eval Loss: {avg_eval_loss:.4f}")
                # train again for the next step
                model.train()

    
                
        # At the end of each epoch, log the average loss for the epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_values.append(avg_epoch_loss)
        logger.info(f"Epoch [{epoch}/{num_epochs-1}] Average Loss: {avg_epoch_loss:.4f}")
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, 0, checkpoint_dir=out_dir, ddp=ddp) # save for start of next epoch



        start_step = 0  # Reset step at the beginning of each new epoch

    # end ddp
    if ddp:
        cleanup()

    # Save the final checkpoint
    save_checkpoint(model, optimizer, scheduler, scaler, epoch, step, checkpoint_dir=out_dir, final_checkpoint=True, ddp=ddp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_hdf5_file',
                        action="store",
                        help='training dataset train.h5 path')
    parser.add_argument('--eval_hdf5_file',
                        action="store",
                        help='evaluation dataset eval.h5 path')
    parser.add_argument('-o', '--out_dir',
                        action="store",
                        help='output directory')
    parser.add_argument('-m', '--model_name',
                        default="facebook/esm2_t33_650M_UR50D",
                        action="store",
                        help='model name')
    parser.add_argument('-e','--epochs', default=3, type=int, metavar='N',
                        action="store",
                        help='number of total epochs to run')
    parser.add_argument('-b','--batch_size', default=32, type=int, metavar='N',
                        action="store",
                        help='batch size')
    parser.add_argument('-t', '--threads', default=1, type=int, metavar='N',
                        action="store",
                        help='number of total threads')
    parser.add_argument('-s','--steps', default=228640, type=int, metavar='N',
                        action="store",
                        help='number of steps per epoch')
    parser.add_argument('--eval_steps', default=500, type=int, metavar='N',
                        action="store",
                        help='number of steps per evaluation loss calculation')
    parser.add_argument('--ddp', 
                        action="store_true",
                        help='whether using DDP')
    args = parser.parse_args()

    train_esm2(args.train_hdf5_file, args.eval_hdf5_file, args.out_dir, args.model_name, args.epochs, args.threads, args.batch_size, args.steps, args.eval_steps, args.ddp)



if __name__ == '__main__':
    main()