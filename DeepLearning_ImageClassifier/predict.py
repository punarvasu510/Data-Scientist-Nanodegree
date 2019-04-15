import argparse
from classpredict import classpredict

parser = argparse.ArgumentParser(description='Predict flower name from the image provided along with the probability of that name using model from provided checkpoint')
parser.add_argument("image_path",help='Path to desired flower image')
parser.add_argument("--checkpoint",help='Path to checkpoint containing the model',nargs='?',default="checkpoint.pth")
parser.add_argument("--top_k",help='Returns top K most likely classes')
parser.add_argument("--category_names",help='Use a mapping of categories to real names',nargs='?',default="cat_to_name.json")
parser.add_argument("--gpu",help='Use GPU for inference',action="store_true")

args=parser.parse_args()
output=classpredict(args.image_path,args.checkpoint,args.top_k,args.category_names,args.gpu)

if output:
    for name,prob in zip(output[0],output[1]):
        print("Flower name: ",name,"\t","Probability: ",prob)
