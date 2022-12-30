import cv2
import torch


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks = None, transform=None, predict=False):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.predict = predict

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = cv2.imread(self.images[item])
        if self.masks is None or self.predict:
            if self.transform:
                image = self.transform(image=image)['image']
                return image
             
        mask = cv2.imread(self.masks[item], 0) // 255

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask
    
if __name__ == '__main__':
    import os
    import glob
    all_imgs = glob.glob('/home/aieson/codes/datasets/buildingSegDataset/train/*.jpg')
    all_masks = [x.replace('train', 'mask') for x in all_imgs]
    # print(all_imgs)
    ds = SegmentationDataset(all_imgs, all_masks)
    print(ds[0][1].max())