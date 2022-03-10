import torch
from torch import nn
from sklearn.metrics import r2_score
import logging
from tqdm import tqdm
from torch.autograd import Variable
import transformers
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

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
            pred = None
            loss = None

            ## perform forward pass
            if config_dict["model_name"] == "siamese_lstm_attention":
                (
                    pred,
                    sent1_annotation_weight_matrix,
                    sent2_annotation_weight_matrix,
                ) = model(
                    sent1.to(device),
                    sent2.to(device),
                    sents1_len.to(device),
                    sents2_len.to(device),
                )
                sent1_attention_loss = attention_penalty_loss(
                    sent1_annotation_weight_matrix,
                    config_dict["self_attention_config"]["penalty"],
                    device,
                )
                sent2_attention_loss = attention_penalty_loss(
                    sent2_annotation_weight_matrix,
                    config_dict["self_attention_config"]["penalty"],
                    device,
                )
                ## compute loss
                loss = (
                    criterion(
                        pred.to(device),
                        torch.autograd.Variable(targets.float()).to(device),
                    )
                    + sent1_attention_loss
                    + sent2_attention_loss
                )
            else:
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
        val_acc, val_loss = evaluate_dev_set(
            model, data, criterion, dataloader, config_dict, device
        )

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


def evaluate_dev_set(model, data, criterion, data_loader, config_dict, device):
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
        pred = None
        loss = None

        ## perform forward pass
        if config_dict["model_name"] == "siamese_lstm_attention":
            (
                pred,
                sent1_annotation_weight_matrix,
                sent2_annotation_weight_matrix,
            ) = model(
                sent1.to(device),
                sent2.to(device),
                sents1_len.to(device),
                sents2_len.to(device),
            )
            sent1_attention_loss = attention_penalty_loss(
                sent1_annotation_weight_matrix,
                config_dict["self_attention_config"]["penalty"],
                device,
            )
            sent2_attention_loss = attention_penalty_loss(
                sent2_annotation_weight_matrix,
                config_dict["self_attention_config"]["penalty"],
                device,
            )
            ## compute loss
            loss = (
                criterion(
                    pred.to(device),
                    torch.autograd.Variable(targets.float()).to(device),
                )
                + sent1_attention_loss
                + sent2_attention_loss
            )
        else:
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


def attention_penalty_loss(annotation_weight_matrix, penalty_coef, device):
    """
    This function computes the loss from annotation/attention matrix
    to reduce redundancy in annotation matrix and for attention
    to focus on different parts of the sequence corresponding to the
    penalty term 'P' in the ICLR paper
    ----------------------------------
    'annotation_weight_matrix' refers to matrix 'A' in the ICLR paper
    annotation_weight_matrix shape: (batch_size, attention_out, seq_len)
    """
    batch_size, attention_out_size = annotation_weight_matrix.size(
        0
    ), annotation_weight_matrix.size(1)
    ## this fn computes ||AAT - I|| where norm is the frobenius norm
    ## taking transpose of annotation matrix
    ## shape post transpose: (batch_size, seq_len, attention_out)
    annotation_weight_matrix_trans = annotation_weight_matrix.transpose(1, 2)

    ## corresponds to AAT
    ## shape: (batch_size, attention_out, attention_out)
    annotation_mul = torch.bmm(annotation_weight_matrix, annotation_weight_matrix_trans)

    ## corresponds to 'I'
    identity = torch.eye(annotation_weight_matrix.size(1))
    ## make equal to the shape of annotation_mul and move it to device
    identity = Variable(
        identity.unsqueeze(0)
        .expand(batch_size, attention_out_size, attention_out_size)
        .to(device)
    )

    ## compute AAT - I
    annotation_mul_difference = annotation_mul - identity

    ## compute the frobenius norm
    penalty = frobenius_norm(annotation_mul_difference)

    ## compute loss
    loss = (penalty_coef * penalty / batch_size).type(torch.FloatTensor)

    return loss


def frobenius_norm(annotation_mul_difference):
    """
    Computes the frobenius norm of the annotation_mul_difference input as matrix
    """
    return torch.sum(
        torch.sum(torch.sum(annotation_mul_difference ** 2, 1), 1) ** 0.5
    ).type(torch.DoubleTensor)
