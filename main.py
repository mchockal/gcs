import copy
import time

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import dataset
import nvidia

LEARNING_RATE = 0.001
MOMENTUM = 0.5
WEIGHT_DECAY_REGULARIZATION_TERM = 0.005
NUM_EPOCHS = 1

# Citation:
# - AverageMeter taken verbatim from the Assignment 2 training code.
# - Remainder of code in this file based on Assignment 2 traiing code.

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(epoch, data_loader, model, optimizer, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()

    for idx, (data, target) in enumerate(data_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        # Forward pass and computation of loss.
        out = model(data)
        out = out.reshape(target.shape)
        loss = criterion(out, target)

        # Backwards pass to determine gradients and update model parameters.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(epoch, idx, len(data_loader), iter_time=iter_time, loss=losses)))

def validate(epoch, validation_loader, model, criterion):
    iter_time = AverageMeter()
    losses = AverageMeter()

    for idx, (data, target) in enumerate(validation_loader):
        start = time.time()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        out = None
        loss = None

        with torch.no_grad():
            out = model(data)
            loss = criterion(out, target)

        losses.update(loss, out.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t')
                  .format(epoch, idx, len(validation_loader), iter_time=iter_time, loss=losses))

    print("* Average Loss @1: {loss.avg:.4f}".format(loss=losses))
    return losses.avg

def main():
    # Normalizing images per the paper and resizing each image to 66 x 200.
    transform = transforms.Compose([
        # Citation:
        # https://pytorch.org/vision/stable/transforms.html#scriptable-transforms
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Resize((66, 200)),
    ])

    # Loading in images with normalization and resizing applied.
    training_set, validation_set, test_set = dataset.load_nvidia_dataset(transform=transform)

    # Loading in the NVIDIA DAVE-2 model.
    model = nvidia.NvidiaDaveCNN()

    # Specifying Mean Squared Error (MSE) as the criterion since this is a regression task.
    criterion = nn.MSELoss()

    # Using Stochastic Gradient Descent (SGD) as the optimizer.
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY_REGULARIZATION_TERM)

    best = 0.0
    best_model = None
    for epoch in range(NUM_EPOCHS):
        # TODO - Consider adding an adjustable learning rate.

        # Training.
        train(epoch, training_set, model, optimizer, criterion)

        # Validation.
        average_loss = validate(epoch, test_set, model, criterion)

        if average_loss > best:
            best = average_loss
            best_model = copy.deepcopy(model)

    print('Best Loss @1: {:.4f}'.format(best))

    torch.save(best_model.state_dict(), './checkpoints/nvidia_dave2.pth')

if __name__ == '__main__':
    main()
