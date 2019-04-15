import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session


def train_model(data_dir,save_dir,arch,gpu,hidden_units,learning_rate,epochs):

    if not save_dir:
        save_dir = "."

    if not learning_rate:
        learning_rate=0.003
    learning_rate = float(learning_rate)

    if not epochs:
        epochs=3
    epochs = int(epochs)

    if gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("Error: GPU not available")
            return
    else:
        print("You are currently using CPU. Do you want to use GPU instead? [y/n]")
        yesno=str(input())
        if yesno=="Y" || yesno=="y":
            device = torch.device("cuda")
        else:
            device="cpu"

    print("Training the model on",device)
    
    # Datasets directory
    #if '/' in data_dir:
    #    delim='/'
    #elif '\\' in data_dir:
    #    delim='\\'
    #else:
    #    return

    delim='/'

    data_dir = data_dir.rstrip(delim)
    train_dir = data_dir + delim + 'train'
    valid_dir = data_dir + delim + 'valid'
    test_dir = data_dir + delim + 'test'

    #Transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    #Loading datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    #Defining dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    #Model Building
    if arch=="vgg19":
        model = models.vgg19(pretrained=True)
    else:
        arch="densenet121"
        model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    if arch=="vgg19":
        hidden_units = hidden_units if hidden_units else 4096
        if hidden_units > 1024:
            model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(hidden_units, 1024),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(1024, 102),
                                        nn.LogSoftmax(dim=1))
        else:
            model.classifier = nn.Sequential(nn.Linear(25088,4096),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(4096,hidden_units),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(hidden_units, 102),
                                        nn.LogSoftmax(dim=1))

    else:
        hidden_units = hidden_units if hidden_units else 512
        model.classifier = nn.Sequential(nn.Linear(1024, hidden_units),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(hidden_units, 102),
                                        nn.LogSoftmax(dim=1))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    model.to(device)

    # Model training and validation
    steps = 0
    running_loss = 0
    print_every = 20
    with active_session():
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)
                            test_loss += batch_loss.item()

                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs} || "
                          f"Train loss: {running_loss/print_every:.3f} || "
                          f"Validation loss: {test_loss/len(validloader):.3f} || "
                          f"Validation accuracy: {accuracy/len(validloader):.3f}")
                    running_loss = 0
                    model.train()

    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'arch': arch,
              'classifier': model.classifier,
              'class_to_idx': model.class_to_idx,
              'epochs': epochs,
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()}
    
    torch.save(checkpoint,save_dir.rstrip(delim)+delim+'checkpoint.pth')