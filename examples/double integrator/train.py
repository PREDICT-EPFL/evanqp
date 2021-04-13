import argparse
import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from evanqp import FFNN
from torch.utils.tensorboard import SummaryWriter

# disable TF32 mode to increase numerical precision
# (makes matrix operations on GPUs slower)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class DoubleIntegratorDataset(Dataset):
    def __init__(self, parameter_samples, variable_samples):
        self.parameter_samples = parameter_samples
        self.variable_samples = variable_samples

    def __len__(self):
        return self.parameter_samples.shape[0]

    def __getitem__(self, idx):
        return self.parameter_samples[idx, :], self.variable_samples[idx, :]


def loss_function(model, data, target):
    output = model(data)
    loss = F.mse_loss(output, target) + 1e2 / len(data) * F.mse_loss(model(torch.zeros(data.shape[1])), torch.zeros(1))
    return loss


def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        loss = loss_function(model, data, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

    train_loss /= len(train_loader)

    if writer is not None:
        writer.add_scalar('train/loss', train_loss, epoch)


def test(model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            test_loss += loss_function(model, data, target).item()

    test_loss /= len(test_loader)

    print('\nTest set: Average loss: {:.6f})\n'.format(test_loss))
    if writer is not None:
        writer.add_scalar('test/loss', test_loss, epoch)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Double Integrator Training Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--train-set-ratio', type=float, default=0.8, metavar='M',
                        help='Portion of dataset used for training (default: 0.8)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.995, metavar='M',
                        help='Learning rate step gamma (default: 0.995)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='save training statistics for tensorboard')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    parameter_samples = torch.load('parameter_samples.pt')
    variable_samples = torch.load('variable_samples.pt')
    dataset = DoubleIntegratorDataset(parameter_samples, variable_samples)
    train_set_size = int(len(dataset) * args.train_set_ratio)
    test_set_size = len(dataset) - train_set_size
    train_set, test_set = random_split(dataset, [train_set_size, test_set_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, **train_kwargs)
    test_loader = DataLoader(test_set, **test_kwargs)

    parameter_size = parameter_samples.shape[1]
    variable_size = variable_samples.shape[1]
    depth = 3
    hidden_size = 25

    model = FFNN([hidden_size for _ in range(depth)], input_size=parameter_size, output_size=variable_size)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if args.tensorboard:
        writer = SummaryWriter('runs/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        writer = None

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(model, device, test_loader, epoch, writer)
        scheduler.step()
        if args.tensorboard:
            writer.flush()
    if args.tensorboard:
        writer.close()

    if args.save_model:
        torch.save({
            'input_size': parameter_size,
            'output_size': variable_size,
            'depth': depth,
            'hidden_size': hidden_size,
            'state_dict': model.state_dict(),
        }, 'double_integrator_ffnn.pt')


if __name__ == '__main__':
    main()
