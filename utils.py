import csv
from glob import glob
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import cm as CM
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

norm = 100

def show_gt(img_path):
  gt_path = img_path.replace("img", "gt").replace(".jpg", ".txt")
  with open(gt_path, "r") as file:
    # Read the file into a string
    file_string = file.read()
    # Split the string into lines
    lines = file_string.split("\n")
    # Count the number of lines (-1 because there is always an empty line)
    gt = len(lines)-1
    print("Number of people (groundtruth):", gt)

  #print(Image.open(img_path).size)
  plt.imshow(Image.open(img_path))
  return gt

def show_map(img_path):
  map_path = img_path.replace("img", "den").replace(".jpg", ".csv")
  print(map_path)
  density_map_csv = pd.read_csv(map_path, header=None)
  density_map = density_map_csv.values
  print(density_map.shape)
  #cmap='gray'
  plt.imshow(density_map, cmap=CM.jet)
  # Count the number of people
  people_count = np.sum(density_map)
  #people_count2 = np.trapz(np.trapz(density_map, dx=1, axis=1), dx=1, axis=0)
  print("Number of people predicted:", people_count)
  return people_count


def get_errors(count_vec):
	count_vec = np.array(count_vec)
	diff = abs(count_vec[:,0] - count_vec[:,1])
	diff_norm = diff/(1+count_vec[:,0])
	mae = format(np.round(diff.mean(),2),'7.1f')
	mse = format(np.round(np.sqrt((diff*diff).mean()),2),'7.1f')
	return mae, mse

def compute_errors(path_dataset, pred_file, mode):

	gt_file = os.path.join(path_dataset, mode,'image_labels.txt')

	gt = {}
	pred = {}

	with open(gt_file, "r") as f:
		lines = f.readlines()

		for line in lines:
			words = line.strip().split(',')
			gt[words[0]] = {}
			gt[words[0]]['count'] = float(words[1])
			gt[words[0]]['weather'] = float(words[3])

	with open(pred_file, "r") as f:
		lines = f.readlines()
		for line in lines:
			words = line.strip().split(',')

			pred[words[0].split('.')[0]] = float(words[1])

	overall = []
	fog = []
	rain = []
	snow = []
	low = []
	med = []
	high = []
	weather = []
	distractor = []
	for key in sorted(gt.keys()):
		if key in pred.keys():
			overall.append([gt[key]['count'] , pred[key]])
			if gt[key]['weather'] == 1 :
				fog.append([gt[key]['count'] , pred[key]])
			if gt[key]['weather'] == 2 :
				rain.append([gt[key]['count'] , pred[key]])
			if gt[key]['weather'] == 3 :
				snow.append([gt[key]['count'] , pred[key]])

			if gt[key]['weather'] == 1 or  gt[key]['weather'] == 2 or gt[key]['weather'] == 3:
				weather.append([gt[key]['count'] , pred[key]])

			if gt[key]['count'] < 50:
				low.append([gt[key]['count'] , pred[key]])
			elif gt[key]['count'] < 500:
				med.append([gt[key]['count'] , pred[key]])
			else:
				high.append([gt[key]['count'] , pred[key]])
		else:
			print('Error: file not found in prediction. Please ensure "mode" is selected appropriately.')
			exit(0)

	mae_overall, mse_overall = get_errors(overall)
	mae_fog, mse_fog = get_errors(fog)
	mae_rain, mse_rain = get_errors(rain)
	mae_snow, mse_snow = get_errors(snow)
	mae_weather, mse_weather = get_errors(weather)
	mae_low, mse_low = get_errors(low)
	mae_med, mse_med = get_errors(med)
	mae_high, mse_high = get_errors(high)

	print(''.ljust(20),'mae_low',',','mse_low',',','mae_med',',','mse_med',',',\
		'mae_high',',','mse_high',',','mae_weather',',','mse_weather',',','mae_overall',',','mse_overall')

	print(pred_file.split('/')[-1].split('.')[0].ljust(20),',',mae_low,',',mse_low,',',mae_med,',',mse_med,',',\
		mae_high,',',mse_high,',',mae_weather.rjust(12),',',mse_weather.rjust(12),',',mae_overall.rjust(10),',',mse_overall.rjust(10))

	return mae_overall, mse_overall

class RandomCrop(object):

   def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = output_size
        else:
            assert len(output_size) == 2
            self.output_size = output_size

   def __call__(self, sample):
        image, den, name = sample['image'], sample['den'], sample['fname']
        w, h = image.size
        new_h, new_w = self.output_size, self.output_size
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        cropped_image = np.asarray(image)[top: top + new_h,
                      left: left + new_w]
        #cropped density map
        cropped_map = den[top: top + new_h,
                      left: left + new_w]
        return {"image": cropped_image, "den": cropped_map, "fname": name}


class ToTensor(object):
    """Convert Image and density map to tensor"""
    def __call__(self, sample):
        image, den, fname = sample['image'], sample['den'], sample['fname']
        image = np.array(image)
        tfms = transforms.Compose([
            transforms.ToTensor()
        ])

        image = tfms(image)
        den = tfms(den)
        return {"image": image, "den": den, "fname": fname}


class Normalize(object):
    #Normalize Image

    #avg [110.6506,  101.2294,   99.0435]
    #stdev [66.9451,   64.1251,   64.1059]

    #avg [0.4340, 0.3970, 0.3884]
    #std [0.2625, 0.2515, 0.2514]

    # [0.410824894905, 0.370634973049, 0.359682112932]
    # [0.278580576181, 0.26925137639, 0.27156367898]

    # [0.4441, 0.4022, 0.3935]
    # [0.2905, 0.2809, 0.2846]
    def __init__(self, mean=[0.4340, 0.3970, 0.3884], std=[0.2625, 0.2515, 0.2514]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        den, fname = sample['den'], sample['fname']
        tfms = transforms.Compose([
            transforms.Normalize(self.mean, self.std)
        ])
        image = tfms(image)
        return {"image": image, "den": den, "fname": fname}


class RandomFlip(object):

    def __init__(self, rnd=0.5):
        self.rnd = rnd

    def __call__(self, sample):
        image, den, fname = sample['image'], sample['den'], sample['fname']
        pil_image = Image.fromarray(image)
        pil_den = Image.fromarray(den)
        if random.random() < self.rnd:
            pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
            pil_den = pil_den.transpose(Image.FLIP_LEFT_RIGHT)
        image = np.array(pil_image)
        den = np.array(pil_den)
        return {"image": image, "den": den, "fname": fname}


class LabelNormalize(object):
    #Normalize the density map
    def __init__(self, norm):
        self.norm = norm

    def __call__(self, sample):
        image, den, fname = sample['image'], sample['den'], sample['fname']
        den = den * self.norm
        return {"image": image, "den": den, 'fname': fname}


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, den, fname = sample['image'], sample['den'], sample['fname']
        tfms = transforms.Compose([
            transforms.Resize(self.size)
        ])
        image = tfms(image)
        return {"image": image, "den": den, 'fname': fname}

class DynamicNormalize(object):
    def __init__(self):
        self.mean = None
        self.std = None

    def __call__(self, sample):
        image, den, fname = sample['image'], sample['den'], sample['fname']
        if self.mean is None or self.std is None:
            # compute mean and std of dataset
            self.mean = torch.mean(image, dim=(1, 2))  # calculate mean for each channel
            self.std = torch.std(image, dim=(1, 2)) + 1e-6    # calculate std for each channel
        # normalize tensor using current mean and std
        normalized_image = transforms.functional.normalize(image, self.mean, self.std)
        self.mean = None
        self.std = None
        return {"image": normalized_image, "den": den, 'fname': fname}

class DynamicNormalizeVal(object):
    def __init__(self):
        self.mean = None
        self.std = None

    def __call__(self, sample):
        image = sample
        if self.mean is None or self.std is None:
            # compute mean and std of dataset
            self.mean = torch.mean(image, dim=(1, 2))  # calculate mean for each channel
            self.std = torch.std(image, dim=(1, 2)) + 1e-6    # calculate std for each channel
        # normalize tensor using current mean and std
        normalized_image = transforms.functional.normalize(image, self.mean, self.std)
        self.mean = None
        self.std = None
        return normalized_image

class ResizeVal(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image = sample
        tfms = transforms.Compose([
            transforms.Resize(self.size)
        ])
        image = tfms(image)
        return image

class ResizeMaxDimension(transforms.Resize):
    def __init__(self, max_dimension):
        self.max_dimension = max_dimension

    def __call__(self, img):
        width, height = img.size
        max_dim = max(width, height)
        if max_dim > self.max_dimension:
            scale_factor = self.max_dimension / max_dim
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            return img.resize((new_width, new_height))
        else:
            return img

class CrowdsDataset(Dataset):
    def __init__(self, path, transform=None):
      self.path = path
      self.images = glob(f"{self.path}/img/*")
      self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
        idx = idx.tolist()
      name = self.images[idx]
      image = Image.open(name)
      image = image.convert('RGB')
      den_name = name.replace("img", "den").replace(".jpg", ".csv")
      density_map = pd.read_csv(den_name, header=None).values.astype(np.float32)
      sample = {'image': image, 'den': density_map, "fname": name}
      if self.transform:
        sample = self.transform(sample)

      return sample
    
train_transform = transforms.Compose(
    [
        RandomCrop(256),
        RandomFlip(),
        LabelNormalize(norm),
        ToTensor(),
        DynamicNormalize()             #normalize dinamically so that the mean is 0 and std is 1
    ]
)

val_transform = transforms.Compose([
        #RandomCrop(256),
        #RandomFlip(),
        #LabelNormalize(norm),
        #Resize(size=(512,512)),
        ToTensor(),
        DynamicNormalize()   
    ]
)

test_transform = transforms.Compose([
    #ResizeVal(size=(512,512)),
    ResizeMaxDimension(max_dimension=512),
    transforms.ToTensor(),
    DynamicNormalizeVal()
])

def single_evaluate(model, image_path, norm):
    image = Image.open(image_path)
    #image = ResizeMaxDimension(max_dimension=512)(image)
    image = test_transform(image).float()
    image = image.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    preds = model(image)
    preds = preds.squeeze(0)
    preds = preds.data.cpu().numpy() / norm
    count = np.around(np.sum(preds))
    if count < 0:
      count = 0
    return {
        "fname": image_path,
        "count": count,
        "prediction": preds[0]
    }

def load_images(image_paths):
    images = [Image.open(image_path) for image_path in image_paths]
    return [test_transform(image).float().unsqueeze(0) for image in images]


def calculate_mae(model,device, folder_path, csv_path, mode, norm):
  model.eval()
  model.to(device)
  with open(csv_path, 'w', newline='') as file:
    cont=0
    writer = csv.writer(file)
    for image in os.listdir(folder_path):
      out=single_evaluate(model,folder_path+'/'+image, norm)
      #rand = random.randint(0, 3100)
      writer.writerow([image,str(out['count'])])
      cont+=1
      if (cont%10 == 0):
        print(cont)