from model import HindiLSTMClassifier
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging

logging.basicConfig(level=logging.INFO)

"""
Script for training the neural network and saving the better models 
while monitoring a metric like accuracy etc
"""


def train_model(model, optimizer, dataloader, data, max_epochs, config_dict):
    device = config_dict["device"]
    criterion = nn.BCELoss()  ## since we are doing binary classification
    max_accuracy = 5e-1
    for epoch in range(max_epochs):

        logging.info("Epoch: {}".format(epoch))
        y_true = list()
        y_pred = list()
        total_loss = 0
        for batch, targets, lengths, raw_data in dataloader["train_loader"]:
            batch, targets, lengths = data.sort_batch(
                batch, targets, lengths
            )  ## sorts the batch wrt the length of sequences

            model.zero_grad()

            pred = model(
                torch.autograd.Variable(batch).to(device), lengths.cpu().numpy()
            )  ## perform forward pass
            pred = torch.squeeze(pred)
            loss = criterion(
                pred.to(device), torch.autograd.Variable(targets.float()).to(device)
            )  ## compute loss

            loss.backward()  ## perform backward pass
            optimizer.step()  ## update weights

            pred_val = pred >= 0.5  ## get predictions
            y_true += list(targets.int().numpy())  ## accumulate targets from batch
            y_pred += list(
                pred_val.data.int().detach().cpu().numpy()
            )  ## accumulate preds from batch
            total_loss += loss  ## accumulate train loss

        acc = accuracy_score(
            y_true, y_pred
        )  ## computing accuracy using sklearn's function

        ## compute model metrics on dev set
        val_acc, val_loss = evaluate_dev_set(model, data, criterion, dataloader, device)

        if val_acc > max_accuracy:
            max_accuracy = val_acc
            logging.info(
                "new model saved"
            )  ## save the model if it is better than the prior best
            torch.save(model.state_dict(), "{}.pth".format(config_dict["model_name"]))

        logging.info(
            "Train loss: {} - acc: {} -- Validation loss: {} - acc: {}".format(
                torch.mean(total_loss.data.float()), acc, val_loss, val_acc
            )
        )
    return model


def evaluate_dev_set(model, data, criterion, data_loader, device):
    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating accuracy on dev set")

    y_true = list()
    y_pred = list()
    total_loss = 0
    for batch, targets, lengths, raw_data in data_loader["dev_loader"]:
        batch, targets, lengths = data.sort_batch(
            batch, targets, lengths
        )  ## sorts the batch wrt the length of sequences

        pred = model(
            torch.autograd.Variable(batch).to(device), lengths.cpu().numpy()
        )  ## perform forward pass
        pred = torch.squeeze(pred)
        loss = criterion(
            pred.to(device), torch.autograd.Variable(targets.float()).to(device)
        )  ## compute loss
        pred_val = pred >= 0.5  ## get predictions
        y_true += list(targets.int())
        y_pred += list(pred_val.data.int().detach().cpu().numpy())
        total_loss += loss

    acc = accuracy_score(y_true, y_pred)  ## computing accuracy using sklearn's function

    return acc, torch.mean(total_loss.data.float())
