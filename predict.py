import argparse
from predict_utils import load_build_model, process_image, imshow, predict, open_cat_to_name

parser = argparse.ArgumentParser(description='Predict Image Classifier')
parser.add_argument('--image','-i', help="Path Image", default='flowers/test/1/image_06743.jpg')
parser.add_argument('--checkpoint','-c', help="Load Path Checkpoint", default='bestmodelcheckpoint1.pth')
parser.add_argument('--arch','-a', help="Choose architecture", default='vgg16')
parser.add_argument('--top_k','-t', help="Return Top K", type=int,default=3)
parser.add_argument('--category_names','-cn', help="Use Category Names", default='cat_to_name.json')
parser.add_argument('--gpu','-g', help="Choose GPU", default='cpu')
    

if __name__ == "__main__":
    args = parser.parse_args()
    model = load_build_model(args.checkpoint, args.gpu, args.arch)
    #image="flowers/test/1/image_06743.jpg"
    #image="flowers/test/10/image_07090.jpg"
    #image="flowers/test/37/image_03734.jpg"
    image=args.image
    probs, classes = predict(image, model, args.gpu, args.top_k)
    cat_to_name = open_cat_to_name(args.category_names) 
    
    for i in range(args.top_k):
        predict_prob = probs[0][i].item()
        predict_class = classes[i]
        print(f'{predict_prob:.2%}: {cat_to_name[str(predict_class)]}')
        


            