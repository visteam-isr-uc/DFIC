from torch.utils.data import Dataset
from PIL import Image



class TONO_Dataset(Dataset):
    def __init__(self, transform=None, df_info = None):
        

        self.transform = transform

        final_info = df_info
        
        self.paths = final_info.loc[:, 'im_path'].to_list()
        self.label = final_info.loc[:, 'label'].to_list()
        self.complete_info = final_info



    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        path = self.paths[idx]
        label = self.label[idx]

       
        im = Image.open(path)


        if self.transform:
            im = self.transform(im)
        


        return im, label, path

