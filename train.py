import os
import sys
import argparse
import torch
from torchtext import data, datasets
import torch.nn.functional as F

from model import ConvText


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=40,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    return p.parse_args()


def train(model, optimizer, train_iter, vocab_size):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text, batch.label
        y.data.sub_(1)  # index align
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()
        if b % 100 == 0:
            corrects = (torch.max(logit, 1)[1].view(
                            y.size()).data == y.data
                        ).sum()
            accuracy = 100.0 * corrects / batch.batch_size
            sys.stdout.write(
                '\rBatch[%d] - loss: %.6f  acc: %.4f (%d/%d)' % (
                 b, loss.data[0], accuracy, corrects, batch.batch_size))


def evaluate(model, val_iter, vocab_size):
    """evaluate model"""
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text, batch.label
        y.data.sub_(1)  # index align
        x, y = x.cuda(), y.cuda()
        logit = model(x)
        loss = F.cross_entropy(logit, y, size_average=False)
        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = avg_loss / size
    accuracy = 100.0 * corrects / size
    return avg_loss, accuracy


def main():
    # get hyper parameters
    args = parse_arguments()
    assert torch.cuda.is_available()

    # load data
    print("\nLoading data...")
    TEXT = data.Field(lower=True, tokenize=list)
    LABEL = data.Field(sequential=False)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    TEXT.build_vocab(train_data, min_freq=5)
    LABEL.build_vocab(train_data)
    train_iter, test_iter = data.BucketIterator.splits(
            (train_data, test_data), batch_size=args.batch_size,
            shuffle=True, repeat=False)
    vocab_size = len(TEXT.vocab)
    n_classes = len(LABEL.vocab) - 1
    print("[TRAIN]: %d \t [TEST]: %d \t [VOCAB] %d \t [CLASSES] %d"
          % (len(train_iter), len(test_iter), vocab_size, n_classes))

    model = ConvText(vocab_size, 128, n_classes, 0.5).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = None
    for e in range(1, args.epochs+1):
        train(model, optimizer, train_iter, vocab_size)
        val_loss, val_accuracy = evaluate(model, test_iter, vocab_size)
        print("\n[Epoch: %d] val_loss:%5.2f | acc:%5.2f" %
              (e, val_loss, val_accuracy))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(model.state_dict(), './.save/convcnn_%d.pt' % (e))
            best_val_loss = val_loss


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP] - training stopped due to interrupt")
