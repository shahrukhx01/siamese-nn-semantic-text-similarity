import torch
from torch import nn
from sklearn.metrics import r2_score
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

"""
Script for training the neural network and saving the better models 
while monitoring a metric like accuracy etc
"""


def train_model(model, optimizer, dataloader, data, max_epochs, config_dict):
    device = config_dict["device"]
    criterion = nn.MSELoss()  ## since we are doing binary classification
    max_accuracy = 5e-1
    for epoch in tqdm(range(max_epochs)):

        logging.info("Epoch: {}".format(epoch))
        y_true = list()
        y_pred = list()
        total_loss = 0
        for (
            sent1,
            sent2,
            sents1_len,
            sents2_len,
            targets,
            _,
            _,
        ) in dataloader["train_loader"]:

            model.zero_grad()
            ## perform forward pass
            pred = model(
                sent1.to(device),
                sent2.to(device),
                sents1_len.to(device),
                sents2_len.to(device),
            )

            ## compute loss
            loss = criterion(
                pred.to(device), torch.autograd.Variable(targets.float()).to(device)
            )

            ## perform backward pass
            loss.backward()

            ## update weights
            optimizer.step()

            ## accumulate targets from batch
            y_true += list(targets.float().numpy())

            ## accumulate preds from batch
            y_pred += list(pred.data.float().detach().cpu().numpy())

            ## accumulate train loss
            total_loss += loss

            # print(y_true, y_pred)

        ## computing accuracy using sklearn's function
        acc = r2_score(y_true, y_pred)

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
    for (
        sent1,
        sent2,
        sents1_len,
        sents2_len,
        targets,
        _,
        _,
    ) in data_loader["val_loader"]:
        ## perform forward pass
        pred = model(
            sent1.to(device),
            sent2.to(device),
            sents1_len.to(device),
            sents2_len.to(device),
        )
        ## compute loss
        loss = criterion(
            pred.to(device), torch.autograd.Variable(targets.float()).to(device)
        )

        y_true += list(targets.float())
        y_pred += list(pred.data.float().detach().cpu().numpy())
        total_loss += loss
    ## computing accuracy using sklearn's function
    acc = r2_score(y_true, y_pred)

    return acc, torch.mean(total_loss.data.float())
