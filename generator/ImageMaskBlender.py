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
# 2024/07/28 ImageMaskBlender.py

import os
import sys
import glob
import cv2
import traceback
import shutil

class ImageMaskBlender:

  def __init__(self):
    self.blur_size=(5,5)
    pass

  def blend(self, images_dir, masks_dir, output_dir):
 
    image_files  = glob.glob(images_dir + "/*.jpg")
    image_files  = sorted(image_files)
   
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_file in image_files:
      basename  = os.path.basename(image_file)
      mask_file = os.path.join(masks_dir, basename)

      name     = basename.split(".")[0]
     
      img      = cv2.imread(image_file)
      mask     = cv2.imread(mask_file)
      #mask     = cv2.blur(mask, self.blur_size)
      img += mask
      merged_file = os.path.join(output_dir, basename)
      cv2.imwrite(merged_file, img)
      print("=== Blended {}".format(merged_file))


if __name__ == "__main__":
  try:
    type = "t2w"
    if len(sys.argv) == 2:
      type = sys.argv[1]
      
    types = ["t1w", "t1c", "t1n", "t2f", "t2w", ]
    if not type in types:  
      error = "Invalid type:"
      raise Exception(error) 
    images_dir = "./Pretreat-MetsToBrain-"+ type + "-master/images"
    masks_dir  = "./Pretreat-MetsToBrain-"+ type + "-master/masks"
    
    output_dir = "./Blended_dataset-" + type
    
    blender = ImageMaskBlender()
    blender.blend(images_dir, masks_dir, output_dir)
  except:
    traceback.print_exc()
