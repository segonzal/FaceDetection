#!/usr/bin/env/python
# -*- coding: utf-8 -*-

import numpy as np

def custom_sampler(num_items, num_epochs, batch_size):
    r = batch_size - num_items % batch_size
    indices = np.vstack([
        np.hstack([
            np.random.permutation(num_items),
            np.random.permutation(num_items)[:r]
        ])
        for _ in range(num_epochs)
    ])
    return indices.reshape(-1, batch_size).astype(np.int32)

class Trainer(object):
    def __init__(self, model, train_dataset, test_dataset=None, num_workers=1, device='cpu'):
        self.model = model
        self.train_dataset = train_dataset
        self.num_workers = num_workers
        self.device = device
        self.epoch_start = 0

    def __call__(self, num_epochs, batch_size):
        num_items   = len(self.train_dataset)
        num_batches = math.ceil(num_items / batch_size)

        step_start = self.epoch_start * num_batches
        sampler = custom_sampler(num_items, num_epochs, batch_size)
        sampler = sampler[step_start:]

        train_dataloader = dd.DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                num_workers=self.num_workers)

        model = self.model.to(self.device)

        # TODO: log num_epochs, batch_size, num_items

        for step, train_batch_data in enumerate(train_loader, step_start):
            epoch = step // num_batches
            batch = step % num_batches

            # TODO: move batch data to device

            # Train
            with torch.enable_grad():
                model.train()
                train_loss = self.train()

            # TODO: log train loss vs step

    def train(self):
        raise NotImplementedError
