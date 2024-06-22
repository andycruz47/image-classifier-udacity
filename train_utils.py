import torch
from torchvision import datasets, transforms, models
from torch import nn, optim

def load_data(batch_size, data_dir):
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([
                            transforms.Resize((224,224)), 
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(10),
                            transforms.RandomResizedCrop(224),
                            transforms.ToTensor(), 
                            transforms.Normalize((0.485,0.456,0.406), std=[0.229,0.224,0.225]) 
                        ])

    test_transforms = transforms.Compose([
                            transforms.Resize((224,224)), 
                            transforms.ToTensor(), 
                            transforms.Normalize((0.485,0.456,0.406), std=[0.229,0.224,0.225]) 
                        ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    batch_size = batch_size

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    return train_dataloader, valid_dataloader, test_dataloader, train_dataset

def train_model(gpu, epochs, learning_rate, train_dataloader, valid_dataloader):
    device = torch.device(gpu)
    model = getattr(models, arch_name)(pretrained=True)
    #model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(25088, 1588),
                                     nn.ReLU(),
                                     nn.Linear(1588, 488),
                                     nn.ReLU(),                                 
                                     nn.Linear(488, 102), 
                                     nn.LogSoftmax(dim=1))

    model.classifier = classifier

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=learning_rate, momentum=0.9)
    

    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for images, labels in train_dataloader:

            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()
            outputs = model(images)
            predicted_exp_prob= torch.exp(outputs)
            _, predicted = predicted_exp_prob.topk(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += (labels==predicted.view(labels.shape)).sum().item()
            #print(running_correct)
        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = 100.00 * running_correct / total

        print("-"*100)
        print(f"Epoch {epoch+1} - Training Dataset: Got {running_correct} Out Of {total} Images Correctly: {round(epoch_acc,2)} - Training loss: {round(epoch_loss,3)}")

        model.eval()
        predicted_correctly_on_epoch=0
        total=0

        with torch.no_grad():
            for images, labels in valid_dataloader:

                images = images.to(device)
                labels = labels.to(device)
                total += labels.size(0)

                outputs = model(images)
                predicted_exp_prob= torch.exp(outputs)
                _, predicted = predicted_exp_prob.topk(1)
                predicted_correctly_on_epoch += (labels==predicted.view(labels.shape)).sum().item()

            epoch_acc = 100.0 * predicted_correctly_on_epoch / total

            print(f"Epoch {epoch+1} - Testing Dataset Got {predicted_correctly_on_epoch} Out Of {total} Images Correctly:   {round(epoch_acc,2)} ")
            
    print("Finish")
    
    return model

def test_model(gpu, model, test_dataloader):
    device = torch.device(gpu)
   
    model.to(device)
    model.eval()

    testing_correct = 0.0
    total = 0

    for images, labels in test_dataloader:

        images = images.to(device)
        labels = labels.to(device)
        total += labels.size(0)

        outputs = model(images)
        predicted_exp_prob= torch.exp(outputs)
        _, predicted = predicted_exp_prob.topk(1)
        testing_correct += (labels==predicted.view(labels.shape)).sum().item()

    accuracy = 100.0 * testing_correct / total

    print(f" Testing Accuracy: {round(accuracy,2)} ")
    
def save_checkpoint(save_dir, model, train_dataset):
    model.class_to_idx = train_dataset.class_to_idx
    state = {
        'model': model.state_dict(),
        'class_to_idx': model.class_to_idx
        }
    torch.save(state, save_dir)
    
    print("Saved")
    return None