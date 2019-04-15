import argparse
from modeltrainer import train_model

parser = argparse.ArgumentParser(description='Train a new network on the data set provided and save the model as a checkpoint')
parser.add_argument("data_dir",help='Path to directory containing the dataset',nargs='?',default="./flowers/")
parser.add_argument("--save_dir",help='Path to directory to save checkpoints')
parser.add_argument("--arch",help='Choose architecture of the deep neural network')
parser.add_argument("--gpu",help='Use GPU for training',action="store_true")
parser.add_argument("--learning_rate",help='Set hyperparameter - learning rate')
parser.add_argument("--hidden_units",help='Set hyperparameter - hidden units')
parser.add_argument("--epochs",help='Set hyperparameter - epochs')


args=parser.parse_args()
train_model(args.data_dir,args.save_dir,args.arch,args.gpu,args.hidden_units,args.learning_rate,args.epochs)
