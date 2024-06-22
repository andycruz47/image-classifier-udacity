from torchvision import transforms, models
import torch
from torch import nn
from PIL import Image
import json

def load_build_model(model_path, device, arch_name): 
    model = getattr(models, arch_name)(pretrained=True)
    #model = models.vgg16(pretrained=True)
       
    classifier = nn.Sequential(nn.Linear(25088, 1588),
                                     nn.ReLU(),
                                     nn.Linear(1588, 488),
                                     nn.ReLU(),                                 
                                     nn.Linear(488, 102), 
                                     nn.LogSoftmax(dim=1))
    model.classifier = classifier
    
    #param_dict = torch.load(model_path, map_location=torch.device(device))
    param_dict = torch.load(model_path)
    
    model.load_state_dict(param_dict['model'])
    model.class_to_idx = param_dict['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    resize_image = Image.open(image).resize((255, 255))
    
    image_transformation = transforms.Compose([
         transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
     
    processed_image = image_transformation(resize_image)
    
    return processed_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    image = process_image(image_path).unsqueeze(dim=0)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        prediction_exp_prob = torch.exp(output) 
        probs, classes = prediction_exp_prob.topk(topk)
        
        idx_to_class = dict((v,k) for k, v in model.class_to_idx.items())
        classes = [p for q, p in idx_to_class.items() if q in classes[0]]
        
    return probs, classes

def open_cat_to_name(file):
    with open(file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

