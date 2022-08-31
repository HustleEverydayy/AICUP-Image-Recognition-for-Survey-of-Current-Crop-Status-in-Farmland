import os, glob, time, sys, getopt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from PIL import Image
import numpy as np


class MyImageDataset(Dataset):
    def __init__(self, img_folder, transform):
        self.transform = transform
        self.img_folder = img_folder
        subpath = os.path.join(img_folder, "*")
        files = glob.glob(subpath)
        self.image_names = files
        self.labels = np.zeros(len(files))

    #The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = Image.open(self.image_names[index])
        if self.transform != None:
            image = self.transform(image)

        return image


def run(modelfile, imgdir, batch_size, outputfile):
    start = time.time()

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),  # ToTensor : [0, 255] -> [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    num_classes = 17
    model = models.efficientnet_b0(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load("fbl_en0e60.pth"))
    model.cuda()
    model.eval()

    test_data = MyImageDataset(imgdir, test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    print(f"Total images={len(test_data)}")

    with open(outputfile, "w") as f:
        # again no gradients needed
        with torch.no_grad():
            total_batch = len(test_data)//batch_size
            for i, batch_images in enumerate(test_loader):
                images = batch_images.cuda()
                outputs = model(images)
                probs_all = torch.nn.functional.softmax(outputs, dim=1)
                for probs in probs_all:
                    str_outputs = [(lambda x: f'{x:0.4f}')(num) for num in probs]
                    print(str_outputs, file=f)

                if (i+1) % 50 == 0:
                    print(f'lter [{i+1}/{total_batch}]')


    end = time.time()
    print(end - start)


def main(argv):
    modelfile = None
    imgdir = 'train_images'
    batches = 256
    outputfile = 'xxx.csv'
    try:
        opts, args = getopt.getopt(argv[:], "h:m:o:d:b:", [
                                   "model=", "ofile=", "imgdir="])
    except getopt.GetoptError:
        print('test_en1.py -m <model> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test_en1.py -m <model> -o <outputfile>')
            sys.exit()
        elif opt in ("-m", "--model"):
            modelfile = arg
            print(arg)
        elif opt in ("-o", "--ofile"):
            outputfile = arg
            print(arg)
        elif opt in ("-d", "--imgdir"):
            imgdir = arg
            print(arg)
        elif opt in ("-b", "--batches"):
            batches = int(arg)
    run(modelfile, imgdir, batches, outputfile)


#cmd command 


if __name__ == "__main__":
    main(sys.argv[1:])