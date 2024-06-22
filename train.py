import argparse
from train_utils import load_data, train_model, test_model, save_checkpoint

parser = argparse.ArgumentParser(description='Image Classifier')
parser.add_argument('--data_dir','-d', help="Data Path", default='flowers')
parser.add_argument('--save_dir','-s', help="Save Path Checkpoint", default='bestmodelcheckpoint1.pth')
parser.add_argument('--arch','-a', help="Choose architecture", default='vgg16')
parser.add_argument('--learning_rate','-lr', help="Learning Rate", type=float, default=0.01)
parser.add_argument('--hidden_units','-hu', help="Hidden Units", type=int, default=256)
parser.add_argument('--epochs','-e', help="Numbers of Epochs", type=int, default=1)
parser.add_argument('--gpu','-g', help="Choose GPU", default='cpu')

    
if __name__ == "__main__":
    args = parser.parse_args()
                    
    train_dataloader, valid_dataloader, test_dataloader, train_dataset = load_data(32, args.data_dir)
    model = train_model(args.gpu, args.epochs, args.learning_rate, train_dataloader, valid_dataloader)
    test_model(args.gpu, model, test_dataloader)
    save_checkpoint(args.save_dir, model, train_dataset)
            
    