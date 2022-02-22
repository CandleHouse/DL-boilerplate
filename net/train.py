import logging
import os
import numpy as np
from models import xxNet
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from data import DatasetLoad

input_dir = ''
label_dir = ''
dir_checkpoint = '../checkpoints/epochs/'  # every epoch
final_checkpoint = '../checkpoints/'  # the last epoch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)  # the number of batch
    tot = 0
    criterion = torch.nn.MSELoss()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            images, labels = batch['image'], batch['label']

            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                label_pred = net(images)

            tot += criterion(label_pred, labels).item()

            pbar.update(images.shape[0])

    # net.train()
    return labels, label_pred, tot / n_val


def train_net(net, device, epochs=5, batch_size=16, lr=1e-4, val_percent=0.2, save_cp=True):
    # 1. Data prepare and divide
    dataset = DatasetLoad(input_dir, label_dir)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
    # - prepare tensorboard
    writer = SummaryWriter(comment=f'_BS_{batch_size}_LR_{lr}_EPOCH_{epochs}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    # 2. Select optimizer and criterion
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.95, weight_decay=0.0005)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, factor=0.5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30], gamma=0.6)

    criterion = torch.nn.MSELoss()  # or customized

    # 3. Train
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='proj') as pbar:
            total_batch = 0
            for batch in train_loader:
                total_batch = total_batch + 1

                images, labels = batch['image'], batch['label']
                images = images.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.float32)

                label_pred = net(images)
                loss = criterion(label_pred, labels)
                epoch_loss += loss.item()
                if epoch > 0:
                    writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])  # update batch size forward
                global_step += 1

        # Save checkpoint for every epoch
        torch.save(net.state_dict(), dir_checkpoint + 'epoch_{}.pth'.format(epoch))

        # epoch loss average
        writer.add_scalar('Epoch Loss/train', epoch_loss / total_batch, epoch)

        # 4. Validation
        val_labels_true, val_labels_pred, val_score = eval_net(net, val_loader, device)
        # scheduler.step(val_score)
        scheduler.step()
        logging.info('Validation MSE: {}'.format(val_score))

        # - tensorboard record
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch)
            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Loss/test', val_score, epoch)
        writer.add_images('val label/true', val_labels_true, epoch)
        writer.add_images('val label/pred', val_labels_pred, epoch)

    # 5. Save checkpoints
    if save_cp:
        torch.save(net.state_dict(), final_checkpoint + f'CP_BS_{batch_size}_LR_{lr}_EPOCH_{epochs}_0222.pth')
        logging.info(f'Checkpoint saved !')


if __name__ == '__main__':
    # -1. Select device
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 0. Select model and initialize
    model = xxNet()
    model.apply(weights_init)
    model.to(device)

    # 1-5 steps in train_net function
    train_net(net=model, epochs=32, batch_size=16, lr=1e-3, device=device, val_percent=0.3)