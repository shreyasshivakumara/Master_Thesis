import numpy as np
import nibabel as nib
import math
import os
from skimage.util import view_as_windows
import skimage.transform as skTrans

class Train_dataset(object):
    def __init__(self, batch_size, overlapping=1):
        self.batch_size = batch_size
        self.data_path = '/home/student_project/Thesis_project/3Dgan-code/temporal_data'
        self.subject_list = []
        for root, dirs, files in os.walk(self.data_path):
                for name in files:
                        if (name == 'image.nii'):
                                 root_split = root.split('/')
                                 self.subject_list.append(root_split[-2] + '/' + root_split[-1])
        #self.subject_list2 = os.listdir('/home/student_project/Thesis_project/Image')
        #self.subject_list = np.delete(self.subject_list, 120)
        self.heigth_patch = 112  # 128
        self.width_patch = 112  # 128
        self.depth_patch = 76  # 92
        self.margin = 16
        self.overlapping = overlapping
        self.num_patches = (math.ceil((224 / (self.heigth_patch)) / (self.overlapping))) * (
            math.ceil((224 / (self.width_patch)) / (self.overlapping))) * (
                               math.ceil((152 / (self.depth_patch)) / (self.overlapping)))

    def mask(self, iteration):
        subject_batch = self.subject_list[iteration * self.batch_size:self.batch_size + (iteration * self.batch_size)]
        subjects_true = np.empty([self.batch_size, 256, 256, 184])
        i = 0
        for subject in subject_batch:
            if subject != 'MRIpartial.csv':
                filename = os.path.join(self.data_path, subject)
                filename = os.path.join(filename, 'mask.nii')
                #filename = os.path.join(filename, 'Mask.nii.gz')
                proxy = nib.load(filename)
                proxy_fdata = proxy.get_fdata()
                proxy_fdata = skTrans.resize(proxy_fdata, (224,224,152), order=1, preserve_range=True)
                proxy = nib.Nifti1Image(proxy_fdata, proxy.affine, proxy.header)
                data = np.array(proxy.dataobj)

                paddwidthr = int((256 - proxy.shape[0]) / 2)
                paddheightr = int((256 - proxy.shape[1]) / 2)
                paddepthr = int((184 - proxy.shape[2]) / 2)

                if (paddwidthr * 2 + proxy.shape[0]) != 256:
                    paddwidthl = paddwidthr + 1
                else:
                    paddwidthl = paddwidthr

                if (paddheightr * 2 + proxy.shape[1]) != 256:
                    paddheightl = paddheightr + 1
                else:
                    paddheightl = paddheightr

                if (paddepthr * 2 + proxy.shape[2]) != 184:
                    paddepthl = paddepthr + 1
                else:
                    paddepthl = paddepthr

                data_padded = np.pad(data,
                                     [(paddwidthl, paddwidthr), (paddheightl, paddheightr), (paddepthl, paddepthr)],
                                     'constant', constant_values=0)
                subjects_true[i] = data_padded
                i = i + 1
        mask = np.empty(
            [self.batch_size * self.num_patches, self.width_patch + self.margin, self.heigth_patch + self.margin,
             self.depth_patch + self.margin, 1])
        i = 0
        for subject in subjects_true:
            patch = view_as_windows(subject, window_shape=(
                (self.width_patch + self.margin), (self.heigth_patch + self.margin), (self.depth_patch + self.margin)),
                                    step=(self.width_patch - self.margin, self.heigth_patch - self.margin,
                                          self.depth_patch - self.margin))
            for d in range(patch.shape[0]):
                for v in range(patch.shape[1]):
                    for h in range(patch.shape[2]):
                        p = patch[d, v, h, :]
                        p = p[:, np.newaxis]
                        p = p.transpose((0, 2, 3, 1))
                        mask[i] = p
                        i = i + 1
        return mask

    def patches_true(self, iteration):
        subjects_true = self.data_true(iteration)
        patches_true = np.empty(
            [self.batch_size * self.num_patches, self.width_patch + self.margin, self.heigth_patch + self.margin,
             self.depth_patch + self.margin, 1])
        i = 0
        for subject in subjects_true:
            patch = view_as_windows(subject, window_shape=(
                (self.width_patch + self.margin), (self.heigth_patch + self.margin), (self.depth_patch + self.margin)),
                                    step=(self.width_patch - self.margin, self.heigth_patch - self.margin,
                                          self.depth_patch - self.margin))
            for d in range(patch.shape[0]):
                for v in range(patch.shape[1]):
                    for h in range(patch.shape[2]):
                        p = patch[d, v, h, :]
                        p = p[:, np.newaxis]
                        p = p.transpose((0, 2, 3, 1))
                        patches_true[i] = p
                        i = i + 1
        return patches_true

    def data_true(self, iteration):
        subject_batch = self.subject_list[iteration * self.batch_size:self.batch_size + (iteration * self.batch_size)]
        subjects = np.empty([self.batch_size, 224, 224, 152])
        i = 0
        for subject in subject_batch:
            if subject != 'MRIpartial.csv':
                filename = os.path.join(self.data_path, subject)
                filename = os.path.join(filename, 'image.nii')
                #filename = os.path.join(filename, 'Image.nii.gz')
                proxy = nib.load(filename)
                proxy_fdata = proxy.get_fdata()
                proxy_fdata = skTrans.resize(proxy_fdata, (224,224,152), order=1, preserve_range=True)
                proxy = nib.Nifti1Image(proxy_fdata, proxy.affine, proxy.header)
                data = np.array(proxy.dataobj)

                paddwidthr = int((256 - proxy.shape[0]) / 2)
                paddheightr = int((256 - proxy.shape[1]) / 2)
                paddepthr = int((184 - proxy.shape[2]) / 2)

                if (paddwidthr * 2 + proxy.shape[0]) != 256:
                    paddwidthl = paddwidthr + 1
                else:
                    paddwidthl = paddwidthr

                if (paddheightr * 2 + proxy.shape[1]) != 256:
                    paddheightl = paddheightr + 1
                else:
                    paddheightl = paddheightr

                if (paddepthr * 2 + proxy.shape[2]) != 184:
                    paddepthl = paddepthr + 1
                else:
                    paddepthl = paddepthr

                data_padded = np.pad(data,
                                     [(paddwidthl, paddwidthr), (paddheightl, paddheightr), (paddepthl, paddepthr)],
                                     'constant', constant_values=0)

                subjects[i] = data_padded[16:240, 16:240, 16:168]  # remove background
                i = i + 1
        return subjects
