import pandas as pd
import os
from PIL import Image
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class TestDateset():
    def __init__(self,gt_path,test_path,transform=None):
        # csv_path = r'D:\resource\DIP\GTSRB_Final_Test_GT\GT.csv'
        # test_path = r'D:\resource\DIP\GTSRB_Final_Test_Images\GTSRB\Final_Test\Images'

        img_datasets = []

        data = pd.read_csv(gt_path)
        labels = data['ClassId'].values

        imgs_name = os.listdir(test_path)
        # print(imgs_name)
        num_img = len(imgs_name)

        for img_name in imgs_name:
            img_path = os.path.join(test_path,img_name)
            # print(img_name)
            img = Image.open(img_path).convert('RGB')
            img_id = int(img_name.split('.')[0])
            label = labels[img_id]

            img_datasets.append((img,label))
            # print(img_name,label)

        # for i in range(0,num_img-1):
        #     img_name = imgs_name[i]
            
        #     label = labels[i]
        #     # print(img_name,label)
        #     img_path = os.path.join(test_path,img_name)
        #     img = Image.open(img_path).convert('RGB')

        #     img_datasets.append((img,int(label)))

        self.img_datasets = img_datasets
        self.num_img = num_img

        if transform is None:
            self.transform = transforms.ToTensor()
        
        else:
            self.transform = transform

    def __len__(self):
        return self.num_img

    def __getitem__(self,index):
        image,label = self.img_datasets[index]
        image = self.transform(image)
        
        return image,label

if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.Resize(50),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    csv_path = r'/opt/mnt/yl/DIP/dataset/GTSRB1/GT.csv'
    test_path = r'/opt/mnt/yl/DIP/dataset/GTSRB1/test'

    testset = TestDateset(csv_path,test_path,transform=train_transform)
    # img,label = testset[0]
    # img = img.permute(1,2,0).numpy()
    # label = int(label)

    testloader = DataLoader(testset,batch_size=5,shuffle=True)

    for x,y in testloader:
        break

    img = x[0].permute(1,2,0).numpy()
    label = int(y[0].numpy())
    # img.save(r'/opt/mnt/yl/DIP/project/img.jpg')
    plt.imshow(img)
    plt.title(label)
    plt.savefig(r'/opt/mnt/yl/DIP/project/img.jpg')


