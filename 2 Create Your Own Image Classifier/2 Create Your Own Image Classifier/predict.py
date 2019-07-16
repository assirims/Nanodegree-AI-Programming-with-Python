import argparse
import torch
import json
from torch.autograd import Variable
import train
from PIL import Image
import numpy as np


def load_checkpoint(checkpoint):
    state = torch.load(checkpoint)

    arch = state['arch']
    lr = float(state['learning_rate'])
    hidden_units = int(state['hidden_units'])

    model, optimizer, criterion = \
        train.build_model(arch, hidden_units, lr)

    model.class_to_idx = state['class_to_idx']
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    return model


def process_image(image):
    size = 224
    width, height = image.size

    if height > width:
        height = int(max(height * size / width, 1))
        width = int(size)
    else:
        width = int(max(width * size / height, 1))
        height = int(size)

    # resized_image = image.resize((width, height))

    x0 = (width - size) / 2
    y0 = (height - size) / 2
    x1 = x0 + size
    y1 = y0 + size
    cropped_image = image.crop((x0, y0, x1, y1))
    np_image = np.array(cropped_image) / 255.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image_array = (np_image - mean) / std
    np_image_array = np_image.transpose((2, 0, 1))

    return np_image_array


def predict(input_path, model, use_gpu, results_to_show, top_k):
    model.eval()
    image = Image.open(input_path)
    np_array = process_image(image)
    tensor = torch.from_numpy(np_array)

    if use_gpu:
        var_inputs = Variable(tensor.float().cuda(), volatile=True)
    else:
        var_inputs = Variable(tensor, volatile=True).float()

    var_inputs = var_inputs.unsqueeze(0)
    output = model.forward(var_inputs)
    ps = torch.exp(output).data.topk(top_k)
    probs = ps[0].cpu() if use_gpu else ps[0]
    classes = ps[1].cpu() if use_gpu else ps[1]
    class_to_idx_inverted = {
        model.class_to_idx[k]: k for k in model.class_to_idx}
    classes_list = list()
    for label in classes.numpy()[0]:
        classes_list.append(class_to_idx_inverted[label])
    return probs.numpy()[0], classes_list


def get_command_line_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str, help='Image file')

    parser.add_argument('checkpoint', type=str, help='Saved model checkpoint')

    parser.add_argument('--top_k', type=int, help='Return the top K most likely classes')

    parser.set_defaults(top_k=1)

    parser.add_argument('--category_names', type=str, help='File of category names')

    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU')

    parser.set_defaults(gpu=False)

    return parser.parse_args()


def main():
    args = get_command_line_args()
    use_gpu = torch.cuda.is_available() and args.gpu
    print("Input file is: {}".format(args.input))
    print("Checkpoint file is: {}".format(args.checkpoint))
    if args.top_k:
        print("Returning {} most likely classes".format(args.top_k))
    if args.category_names:
        print("Category names file: {}".format(args.category_names))
    if use_gpu:
        print("Using GPU.")
    else:
        print("Using CPU.")

    model = load_checkpoint(args.checkpoint)
    print("Checkpoint loaded.")

    if use_gpu:
        model.cuda()

    if args.category_names:
        with open(args.category_names, 'r') as f:
            categories = json.load(f)
            print("Category names loaded")

    results_to_show = args.top_k if args.top_k else 1

    print("Processing image")
    probabilities, classes = predict(args.input, model, use_gpu, results_to_show, args.top_k)

    if results_to_show > 1:
        print("Top {} Classes for '{}':".format(len(classes), args.input))

        if args.category_names:
            print("{:<30} {}".format("Flower", "Probability"))
            print("------------------------------------------")
        else:
            print("{:<10} {}".format("Class", "Probability"))
            print("----------------------")

        for i in range(0, len(classes)):
            if args.category_names:
                print("{:<30} {:.2f}".format(categories[classes[i]], probabilities[i]))
            else:
                print("{:<10} {:.2f}".format(classes[i], probabilities[i]))
    else:
        print("The most likely class is '{}': probability: {:.2f}".format(categories[classes[0]] if args.category_names else classes[0], probabilities[0]))


if __name__ == "__main__":
    main()
