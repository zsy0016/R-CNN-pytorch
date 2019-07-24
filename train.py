import argparse
import os

import numpy as np
import scipy.io as sio
import torch
import torch.distributed as dist
import torchvision
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

import cnn
from dataset import Flower17

torch.multiprocessing.set_start_method('spawn')


def train(model, train_loader, optimizer, loss_func, args):
    model.train()
    num_samples = 0
    loss_all = 0
    accuracy = 0
    for img, y_fact in train_loader:
        if args.cuda:
            img = img.cuda()
            y_fact = y_fact.cuda()
        optimizer.zero_grad()
        y_pred = model(img)
        loss = loss_func(y_pred, y_fact)
        loss.backward()
        optimizer.step()
        num_samples += len(img)
        loss_all += loss.item() * len(img)
        accuracy += (torch.argmax(y_pred.detach(), dim=-1)
                     == y_fact).sum().item()
    return model, loss_all / num_samples, accuracy / num_samples


def test(model, valid_loader, loss_func, args):
    model.eval()
    num_samples = 0
    loss_all = 0
    accuracy = 0
    for img, y_fact in valid_loader:
        if args.cuda:
            img = img.cuda()
            y_fact = y_fact.cuda()
        y_pred = model(img)
        loss = loss_func(y_pred, y_fact)
        num_samples += len(img)
        loss_all += loss.item() * len(img)
        accuracy += (torch.argmax(y_pred.detach(), dim=-1)
                     == y_fact).sum().item()
    return model, loss_all / num_samples, accuracy / num_samples


def main():
    parser = argparse.ArgumentParser("argparser for R-CNN")
    parser.add_argument("--image_dir", type=str,
                        default="data/17flowers/jpg")
    parser.add_argument("--ground_truth_dir", type=str,
                        default="data/trimaps")
    parser.add_argument("--regions_path", type=str,
                        default="data/17flowers/regions.pkl")
    parser.add_argument("--splits_path", type=str,
                        default="data/datasplits.mat")
    parser.add_argument("--save_dir", type=str, default="model")
    parser.add_argument("--cnn", type=str, default="alexnet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--init_method", type=str,
                        default="file:////home/zhangshuyuan/python/rcnn/tmp/tmp")
    parser.add_argument("--num_replicas", type=int, default=2)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    dist.init_process_group(backend="nccl",
                            init_method=args.init_method,
                            rank=args.rank,
                            world_size=args.num_replicas)
    torch.cuda.set_device(args.local_rank)
    assert torch.distributed.is_initialized()

    transform = transforms.Compose([transforms.RandomVerticalFlip(),
                                    transforms.RandomRotation(30),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5))
                                    ])
    train_dataset = Flower17(args.image_dir, args.ground_truth_dir,
                             args.regions_path, args.splits_path, mode='trn', transform=transform)
    valid_dataset = Flower17(args.image_dir, args.ground_truth_dir,
                             args.regions_path, args.splits_path, mode='val')
    if args.cuda:
        train_datasampler = DistributedSampler(
            train_dataset, num_replicas=args.num_replicas, rank=args.rank)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_datasampler)
        valid_datasampler = DistributedSampler(
            valid_dataset, num_replicas=args.num_replicas, rank=args.rank)
        valid_loader = DataLoader(
            valid_dataset, batch_size=args.batch_size, sampler=valid_datasampler)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(
            valid_dataset, batch_size=args.batch_size)

    assert args.cnn in ["alexnet", "vgg"], "unknown cnn model"
    if args.cnn == "alexnet":
        model = cnn.Alexnet()
    if args.cnn == "vgg":
        model = cnn.Vgg()

    if args.cuda:
        model.cuda()
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        print("Epoch %d:" % epoch)
        model, train_loss, train_accuracy = train(
            model, train_loader, optimizer, loss_func, args)
        print("Train: loss=%.4f, accuracy=%.4f" % (train_loss, train_accuracy))
        model, test_loss, test_accuracy = test(
            model, valid_loader, loss_func, args)
        print("Valid: loss=%.4f, accuracy=%.4f" % (test_loss, test_accuracy))

    save_path = os.path.join(args.save_dir, "%s.pkl" % args.cnn)
    torch.save(model.module.state_dict(), save_path)


if __name__ == "__main__":
    main()
