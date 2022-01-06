import torch
from tqdm import tqdm


class FlatTrainer:
    def __init__(self, model, input_shape, optimizer, criterion, device):
        self.model = model
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def fit(self, train_dataset, epochs, batch_size, shuffle=True, verbose=True):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        for epoch in range(epochs):
            loss = self._train_one_epoch(train_loader)
            if verbose: print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))

    def _train_one_epoch(self, train_loader):
        with tqdm(train_loader) as tepoch:
            loss = 0
            for batch_features in tepoch:
                loss += self._train_one_step(batch_features)
        return loss / len(train_loader)

    def _train_one_step(self, batch_features):
        batch_features = batch_features.view(-1, self.input_shape).to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(batch_features)
        train_loss = self.criterion(outputs, batch_features)
        train_loss.backward()
        self.optimizer.step()
        return train_loss.item()


class FlatPredictor:
    def __init__(self, model, input_shape, criterion, device):
        self.model = model
        self.input_shape = input_shape
        self.criterion = criterion
        self.device = device

    def predict(self, dataset, batch_size=10, shuffle=False):
        losses = []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        with torch.no_grad():
            for batch_features in tqdm(dataloader):
                batch_features = batch_features.view(-1, self.input_shape).to(self.device)
                reconstruction = self.model(batch_features)
                reconstruction_error = self.criterion(reconstruction, batch_features)
                losses.append(reconstruction_error)
        return losses