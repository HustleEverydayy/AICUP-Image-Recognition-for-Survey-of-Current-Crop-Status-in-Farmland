import PIL
import os
import sys, getopt

def run(modelfile = '', imgdir = 'train_images', epochs = 1, batches = 8, outputfile = 'TBL_EN0E1.h5'):

    import torch
    import torchvision
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import datasets, transforms
    from torchvision.transforms import ToTensor

    import torchvision.models as models
    num_classes = 17
    model=models.efficientnet_b0(pretrained=False, num_classes=num_classes)

    if modelfile != None:
        model.load_state_dict(torch.load("fbl_en0e60.pth"))

    model.cuda()

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(imgdir, train_transform)
    train_loader = DataLoader(train_data,batch_size=batches, shuffle=True)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = epochs
    for epoch in range(num_epochs):
        total_batch = len(train_data)//batches
        for i, (batch_images, batch_labels) in enumerate(train_loader):
            # Zero your gradients for every batch!
            optimizer.zero_grad()
            inputs = batch_images.cuda()
            labels = batch_labels.cuda()
            # Make predictions for this batch
            outputs  = model(inputs)
            
            # Compute the loss and its gradients
            cost = loss(outputs , labels)
            cost.backward()
            # Adjust learning weights
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], lter [{i+1}/{total_batch}] Loss: {cost.item():.4f}')
    torch.save(model.state_dict(), outputfile)


def main(argv):
   modelfile = None
   imgdir = 'train_images'
   epochs = 1
   batches = 32
   outputfile = 'fbl_pt22-xxx.pth'
   try:
      opts, args = getopt.getopt(argv[:],"h:m:o:d:e:b:",["model=","ofile=", "imgdir=","epochs="])
   except getopt.GetoptError:
      print('train_tf1.py -m <model> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('train_tf1.py -m <model> -o <outputfile>')
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
      elif opt in ("-e", "--epochs"):
         epochs = int(arg)
      elif opt in ("-b", "--batches"):
         batches = int(arg)
   run(modelfile, imgdir, epochs, batches, outputfile)

if __name__ == "__main__":
   main(sys.argv[1:])