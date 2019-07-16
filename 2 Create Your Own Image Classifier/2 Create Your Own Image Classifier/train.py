import os
import argparse
import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
from collections import OrderedDict
from torchvision import datasets, transforms, models

OUTPUT_SIZE = 102
PRINT_EVERY = 40


def get_loaders(data_dir):
    train_dir = data_dir + '/train/'
    valid_dir = data_dir + '/valid/'
    test_dir = data_dir + '/test/'

    data_transforms = {
        'training' : transforms.Compose([transforms.RandomResizedCrop(224),  transforms.RandomHorizontalFlip(), transforms.RandomRotation(30), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        'validation' : transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        'testing' : transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    image_datasets = {
        'training' : datasets.ImageFolder(train_dir, transform=data_transforms['training']),
        'testing' : datasets.ImageFolder(test_dir, transform=data_transforms['testing']),
        'validation' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])}

    dataloaders = {
        'training' : torch.utils.data.DataLoader(image_datasets['training'], batch_size=64, shuffle=True),
        'testing' : torch.utils.data.DataLoader(image_datasets['testing'], batch_size=64, shuffle=False),
        'validation' : torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64, shuffle=True)}

    class_to_idx = image_datasets['training'].class_to_idx
    return dataloaders, class_to_idx


def get_model(arch, hidden_units):
    if arch == 'vgg13_bn':
        model = models.vgg13_bn(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
    elif arch == 'densenet161':
        model = models.densenet161(pretrained=True)
        input_size = model.classifier.in_features
    elif arch == 'densenet201':
        model = models.densenet201(pretrained=True)
        input_size = model.classifier.in_features
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_size = model.fc.in_features
    elif arch == 'resnet34':
        model = models.resnet34(pretrained=True)
        input_size = model.fc.in_features
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        input_size = model.fc.in_features
    else:
        raise Exception("Unknown model")

    for param in model.parameters():
        param.requires_grad = False

    output_size = OUTPUT_SIZE

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    if 'vgg' in arch:
        model.classifier = classifier
    elif 'densenet' in arch:
        model.classifier = classifier
    elif 'resnet' in arch:
        model.fc = classifier

    return model


def build_model(arch, hidden_units, learning_rate):
    model = get_model(arch, hidden_units)
    print("Retrieving the pretrained model")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)
    optimizer.zero_grad()
    criterion = nn.NLLLoss()
    return model, optimizer, criterion


def train_model(model, epochs, criterion, optimizer, train_loader, val_loader, use_gpu):
    print("Starting model training")
    model.train()
    print_every = PRINT_EVERY
    steps = 0

    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in iter(train_loader):
            steps += 1
            if use_gpu:
                inputs = Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)

            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            if steps % print_every == 0:
                val_loss, val_accuracy = validate(model, criterion, val_loader, use_gpu)
                model.train()
                print("Epoch: {}/{} ".format(epoch+1, epochs),
                      "Train Loss: {:.3f} ".format(running_loss/print_every),
                      "Val. Loss: {:.3f} ".format(val_loss),
                      "Val. Acc.: {:.3f}".format(val_accuracy))
                running_loss = 0


def validate(model, criterion, data_loader, use_gpu):
    print("Validating the model")
    model.eval()
    accuracy = 0
    test_loss = 0
    for inputs, labels in iter(data_loader):
        if use_gpu:
            inputs = Variable(inputs.float().cuda(), volatile=True)
            labels = Variable(labels.long().cuda(), volatile=True)
        else:
            inputs = Variable(inputs, volatile=True)
            labels = Variable(labels, volatile=True)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).data[0]
        ps = torch.exp(output).data
        measure = (labels.data == ps.max(1)[1])
        accuracy += measure.type_as(torch.FloatTensor()).mean()

    return test_loss/len(data_loader), accuracy/len(data_loader)


def get_command_line_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str, help='Directory of flower images')

    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Train with GPU')

    parser.set_defaults(gpu=False)

    architectures = {'densenet121', 'densenet161', 'densenet201', 'resnet18', 'resnet34', 'resnet50', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn'}

    parser.add_argument('--save_dir', type=str, help='Directory to save checkpoints')

    parser.add_argument('--arch', dest='arch', default='densenet121', action='store', choices=architectures, help='Architecture to use')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='Model learning rate')

    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')

    parser.add_argument('--epochs', type=int, default=8, help='Number of epochs to train')

    return parser.parse_args()


def save(arch, learning_rate, hidden_units, epochs, save_path, model, optimizer):
    state = {
        'arch': arch,
        'learning_rate': learning_rate,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx}

    torch.save(state, save_path)


def main():
    args = get_command_line_args()
    use_gpu = torch.cuda.is_available() and args.gpu
    print("Data directory is: {}".format(args.data_dir))

    if use_gpu:
        print("Training on GPU.")
    else:
        print("Training on CPU.")

    print("Architecture is: {}".format(args.arch))

    if args.save_dir:
        print("Checkpoint save directory is: {}".format(args.save_dir))

    print("Learning rate is: {}".format(args.learning_rate))
    print("Hidden units are: {}".format(args.hidden_units))
    print("Epochs: {}".format(args.epochs))

    dataloaders, class_to_idx = get_loaders(args.data_dir)

    for key, value in dataloaders.items():
        print("{} data loader retrieved".format(key))

    model, optimizer, criterion = build_model(args.arch, args.hidden_units, args.learning_rate)
    model.class_to_idx = class_to_idx

    if use_gpu:
        print("GPU is availaible.")
        model.cuda()
        criterion.cuda()

    train_model(model, args.epochs, criterion, optimizer, dataloaders['training'], dataloaders['validation'], use_gpu)

    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        save_path = args.save_dir + '/' + args.arch + '_checkpoint.pth'
    else:
        save_path = args.arch + '_checkpoint.pth'
    print("Checkpoint saved to {}".format(save_path))

    save(args.arch, args.learning_rate, args.hidden_units, args.epochs, save_path, model, optimizer)
    print("Checkpoint saved")

    test_loss, accuracy = validate(model, criterion, dataloaders['testing'], use_gpu)
    print("Test Loss: {:.3f}".format(test_loss))
    print("Test Acc.: {:.3f}".format(accuracy))


if __name__ == "__main__":
    main()
