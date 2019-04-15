from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms, models
import json

def load_checkpoint(filepath):
    checkpoint= torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    resize_and_crop = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor()])
    new=np.array(resize_and_crop(image))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    new = (np.transpose(new, (1, 2, 0)) - mean)/std
    new = np.transpose(new, (2, 0, 1))

    return new

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def classpredict(image_path,checkpoint,topk,category_names,gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("Error: GPU not available")
            return
    else:
        print("Use GPU!!")
        return
    
    im = process_image(Image.open(image_path))

    model=load_checkpoint(checkpoint)

    if not topk:
        topk=1
    topk = int(topk)

    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
 
    model.to(device)
    
    #print(image_path,checkpoint,topk,category_names,gpu)
    #print(device)

    image = torch.FloatTensor(im).to(device)
    probs=None
    pred_class=None
    
    model.eval()
    with torch.no_grad():
        image.unsqueeze_(0)
        logps=model.forward(image)
        ps = torch.exp(logps)
        probs, classes = ps.topk(topk, dim=1)

        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        probs=probs.cpu().numpy()[0].tolist()
        classes=classes.cpu().numpy()[0].tolist()
        pred_labels = [idx_to_class[x] for x in classes]
        pred_class = [cat_to_name[str(x)] for x in pred_labels]
        
        #print("probs-----",probs)
        #print("classes-----",pred_class)
    
        return (pred_class,probs)
 