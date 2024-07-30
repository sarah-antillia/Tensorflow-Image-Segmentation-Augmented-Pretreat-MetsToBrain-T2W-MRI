# Copyright 2024 antillia.com Toshiyuki A. Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2024/07/28 ImageMaskDatasetGenerator.py

import os
import sys
import io
import shutil
import glob
import nibabel as nib
import numpy as np
from PIL import Image, ImageOps
import traceback
import matplotlib.pyplot as plt

import cv2


class ImageMaskDatasetGenerator:

  def __init__(self, input_dir="./BraTS21/", type="t2w", output_dir="./BraTS21-master", 
               image_normalize    = True, 
               exclude_empty_mask = True,
               rotation_angle     = 90, 
               resize = 512):
    self.input_dir = input_dir
    self.image_normalize = image_normalize
    self.exclude_empty_mask = exclude_empty_mask

    if not os.path.exists(self.input_dir):
      raise Exception("Not found " + input_dir)   

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    self.output_images_dir = os.path.join(output_dir, "images")
    self.output_masks_dir  = os.path.join(output_dir, "masks")

    if not os.path.exists(self.output_images_dir):
      os.makedirs(self.output_images_dir)

    if not os.path.exists(self.output_masks_dir):
      os.makedirs(self.output_masks_dir)
    self.angle = rotation_angle
    self.BASE_INDEX = 1000

    self.SEG_EXT   = "-seg.nii.gz"
    self.FLAIR_EXT = "-" + type + ".nii.gz"
    self.RESIZE    = (resize, resize)


  def generate(self):
    subdirs = os.listdir(self.input_dir)
    for subdir in subdirs:
      subdir_fullpath = os.path.join(self.input_dir, subdir)
      print("=== subdir {}".format(subdir_fullpath))
      
      seg_files   = glob.glob(subdir_fullpath + "/BraTS-MET*" + self.SEG_EXT)

      flair_files = glob.glob(subdir_fullpath + "/BraTS-MET*" + self.FLAIR_EXT)
      for seg_file in seg_files:
        self.generate_mask_files(seg_file) 
      for flair_file in flair_files:
        self.generate_image_files(flair_file) 
     
  def normalize(self, image):
    min = np.min(image)/255.0
    max = np.max(image)/255.0
    scale = (max - min)
    image = (image -min) / scale
    image = image.astype('uint8') 
    return image
  
  def generate_image_files(self, niigz_file):
    nii = nib.load(niigz_file)
    basename = os.path.basename(niigz_file) 
    nameonly = basename.replace(self.FLAIR_EXT, "")
    fdata  = nii.get_fdata()
   
    w, h, d = fdata.shape
    print("shape {}".format(fdata.shape))
    for i in range(d):
      img = fdata[:,:, i]
      filename  = nameonly + "_" + str(i+self.BASE_INDEX) + ".jpg"
      filepath  = os.path.join(self.output_images_dir, filename)
      corresponding_mask_file = os.path.join(self.output_masks_dir, filename)
      if os.path.exists(corresponding_mask_file):
        if self.image_normalize:
          img   = self.normalize(img)
        image = Image.fromarray(img)
        image = image.convert("RGB")
        image = image.resize(self.RESIZE)
        if self.angle>0:
          image = image.rotate(self.angle)
        image.save(filepath)
        print("=== Saved {}".format(filepath))
      else:
        print("=== Skipped {}".format(filepath))
     
  def generate_mask_files(self, niigz_file ):
    nii = nib.load(niigz_file)
    fdata  = nii.get_fdata()
    w, h, d = fdata.shape
    print("shape {}".format(fdata.shape))
    for i in range(d):
      img = fdata[:,:, i]
      basename = os.path.basename(niigz_file) 
      nameonly = basename.replace(self.SEG_EXT, "")
      if self.exclude_empty_mask:
        if not img.any() >0:
          print("=== Skipped empty mask")
          continue

      img = img*255.0
      img = img.astype('uint8')

      image = Image.fromarray(img)
      image = image.convert("RGB")
      image = image.resize(self.RESIZE)
        
      if self.angle >0:
        image = image.rotate(self.angle)
      filename  = nameonly + "_" + str(i+ self.BASE_INDEX) + ".jpg"
      filepath  = os.path.join(self.output_masks_dir, filename)
      image.save(filepath, "JPEG")
      print("--- Saved {}".format(filepath))

if __name__ == "__main__":
  try:
    input_dir  = "./Pretreat-MetsToBrain-Masks"
    # Default type = "t2w"
    type = "t2w"
    if len(sys.argv) == 2:
      type = sys.argv[1]
      
    types = ["t1w", "t1c", "t1n", "t2f", "t2w", ]
    if not type in types:  
      error = "Invalid type:"
      raise Exception(error)
    
    output_dir = "./Pretreat-MetsToBrain-"+ type + "-master"

    # Enabled image_normalize flag 
    image_normalize = True

    # Enabled exclude_empty_mask     
    exclude_empty_mask = True
    
    # Rotation angle for images and masks
    rotation_angle = 90

    # Resize pixel value
    resize         = 512
    generator = ImageMaskDatasetGenerator(input_dir=input_dir, type=type, output_dir=output_dir, 
                                          image_normalize = image_normalize, 
                                          exclude_empty_mask = exclude_empty_mask,
                                          rotation_angle  = rotation_angle,
                                          resize = resize)
    generator.generate()
  except:
    traceback.print_exc()

 
