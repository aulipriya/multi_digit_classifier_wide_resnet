import argparse
from data_preparation.dataset import Dataset
import torch
from torch.optim.lr_scheduler import MultiStepLR
from model.custom_loss import Custom_loss
from torchvision import transforms
from model.wide_resnet import WideResNet
import csv


def calculate_accuracy(pred_lengths, pred_digit1, pred_digit2, pred_digit3, pred_digit4, pred_digit5, lengths, labels):
    num_correct = 0
    length_prediction = pred_lengths.max(1)[1]
    digit1_prediction = pred_digit1.max(1)[1]
    digit2_prediction = pred_digit2.max(1)[1]
    digit3_prediction = pred_digit3.max(1)[1]
    digit4_prediction = pred_digit4.max(1)[1]
    digit5_prediction = pred_digit5.max(1)[1]

    num_correct += (length_prediction.eq(lengths) &
                    digit1_prediction.eq(labels[0]) &
                    digit2_prediction.eq(labels[1]) &
                    digit3_prediction.eq(labels[2]) &
                    digit4_prediction.eq(labels[3]) &
                    digit5_prediction.eq(labels[4])).cpu().sum()
    accuracy = num_correct.item() / labels.shape[0]
    return accuracy


def validation(model, path_to_test_lmdb, batch_size, loss_object):
    model.eval()
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])

    # Compose transformations to be applied on image
    test_transform = transforms.Compose([])
    test_transform.transforms.append(transforms.Resize((54, 54)))
    test_transform.transforms.append(transforms.ToTensor())
    test_transform.transforms.append(normalize)
    test_dataset = Dataset(path_to_test_lmdb, test_transform)

    test_loader = torch.utils.data.DataLoader(datadet=test_dataset, batch_size=batch_size, shuffle=False)

    total_validation_loss = 0
    total_accuracy = 0
    print('Starting validation')

    for i, data in enumerate(test_loader):
        images, lengths, labels = data
        if torch.cuda.is_available():
            images = images.cuda()
            lengths = lengths.cuda()
            labels = labels.cuda()

        pred_lengths, pred_digit1, pred_digit2, pred_digit3, pred_digit4, pred_digit5 = model(images)
        validation_loss = loss_object.loss(pred_lengths, pred_digit1, pred_digit2, pred_digit3, pred_digit4, pred_digit5,
                                           lengths, labels)/batch_size

        validation_accuracy = calculate_accuracy(pred_lengths, pred_digit1, pred_digit2, pred_digit3, pred_digit4,
                                                 pred_digit5, lengths, labels)
        total_validation_loss += validation_loss
        total_accuracy += validation_accuracy

    total_validation_loss = total_validation_loss / len(test_loader)
    total_accuracy = total_accuracy / len(test_loader)

    return total_validation_loss, total_accuracy


def train_model(model, args):
    # Image normalization
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])

    # Compose transformations to be applied on image
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.Resize((54, 54)))
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    # Define train data set and data loader
    train_dataset = Dataset(args.train_lmdb_path, train_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    # Initialize loss, optimizer, scheduler
    loss_object = Custom_loss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=0.0005)

    scheduler = MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

    best_accuracy = 0
    training_accuracies = dict()
    training_losses = dict()
    validation_accuracies = dict()
    validation_losses = dict()

    # Start training
    for epoch in range(args.epochs):
        total_epoch_loss = 0
        total_epoch_accuracy = 0
        print(f'Starting training for epoch {epoch}')
        for i, data in enumerate(train_loader):
            print(f'Starting training for iteration {i}')
            images, lengths, labels = data
            if torch.cuda.is_available():
                images = images.cuda()
                lengths = lengths.cuda()
                labels = labels.cuda()
                model.cuda()
                loss_object = loss_object.cuda()

            pred_length, pred_digit1, pred_digit2, pred_digit3, pred_digit4, pred_digit5 = model(images)

            loss = loss_object.loss(pred_length, pred_digit1, pred_digit2, pred_digit3, pred_digit4, pred_digit5,
                                    lengths, labels)/args.batch_size
            total_epoch_loss += loss

            total_epoch_accuracy += calculate_accuracy(pred_length, pred_digit1, pred_digit2, pred_digit3, pred_digit4,
                                                       pred_digit5, lengths, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_accuracies[epoch] = total_epoch_accuracy/len(train_loader)
        training_losses[epoch] = total_epoch_loss/len(train_loader)

        validation_loss, validation_accuracy = validation(model, args.val_lmdb_path, args.batch_size, loss_object)

        validation_losses[epoch] = validation_loss
        validation_accuracies[epoch] = validation_accuracy

        if validation_accuracy > best_accuracy:
            torch.save(model.state_dict(), args.weights_path+'/'+epoch+'.pt')

        scheduler.step(epoch)
    return training_losses, training_accuracies, validation_accuracies, validation_losses


def main(args):
    model = WideResNet(depth=16, widen_factor=8, drop_rate=0.4)

    training_losses, training_accuracies, validation_accuracies, validation_losses = train_model(model, args)

    with open('logs.csv', 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train_accuracy', 'train_loss', 'val_accuracy', 'val_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for epoch in range(args.epochs):
            row = {'epoch': str(epoch), 'train_accuracy': str(training_accuracies[epoch]), 'train_loss': str(
                training_losses[epoch]), 'val_accuracy': str(validation_accuracies[epoch]), 'val_loss': str(
                validation_losses[epoch])}
            writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument('--train_lmdb_path',  help='path to train.lmdb directory')
    parser.add_argument('--val_lmdb_path',  help='path to val.lmdb directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--weights_path', default='../weights')
    main(parser.parse_args())
