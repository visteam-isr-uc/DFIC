

from torch.utils.data import Dataset
from PIL import Image




class ICAODataset(Dataset):
    def __init__(self, transform=None, df_info = None, pre_path = '../data/DFIC/preprocessed/'):
        

        self.transform = transform
        self.pre_path = pre_path
        self.paths = df_info.loc[:, 'im_path'].to_list()



    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):


        path = self.paths[idx]

        im = Image.open(self.pre_path + path)
        
        if self.transform:
            im = self.transform(im)


        return im, path