import gc
import os
import nibabel as nib
import models
import numpy as np
import scipy
import matplotlib.pyplot as plt

def load_imgs(dir):
    raw_t1 = nib.load(os.path.join(dir, 'T1.nii'))
    affine = raw_t1._affine
    raw_t1 = raw_t1.get_fdata()
    raw_t2 = nib.load(os.path.join(dir, 'T2_FLAIR.nii')).get_fdata()
    gt = nib.load(os.path.join(dir, 'LabelsForTesting.nii')).get_fdata()
    mask = gt!=0
    return raw_t1, raw_t2, mask, np.asarray(gt), affine

def report_dices(scores):
    classes = {2: 'grey matter', 3: 'white matter', 1: 'csf'}
    for i in range(len(scores)):
        print('Image: ', i+1)
        for k in range(len(scores[i])):
            print(f"Dice for class label {classes[k+1]} is {scores[i][k]:.4f}")

def mean_dice(t1):
    print("COMPUTING MEAN STD DICE")
    classes = {1: 'grey matter', 2: 'white matter', 3: 'csf', 4: 'bone', 5: 'soft-tissue', 6: 'air'}
    t1 = np.asarray(t1)
    for i in range (3):
        std_1 = np.std(t1[:,i])
        mean_1 = np.mean(t1[:,i])
        print(f"Tissue {classes[i+1]} obtained from T1: Mean {mean_1:.4f}, Std {std_1:.4f}")

def load_nifty_as_np(path):
    img = nib.load(path)
    arr = img.get_fdata()
    affine = img._affine
    return arr, affine

def dice(segmentation, gt):
    labels = np.unique(segmentation)
    classes = {1:'grey matter', 2:'white matter', 3:'csf', 4:'bone', 5:'soft-tissue', 6:'air'}
    Dices = []
    for elem in labels:
        if elem == 0:
            continue
        i = np.sum(np.bitwise_and(segmentation == elem, gt==elem))
        u = np.sum(segmentation == elem) + np.sum(gt==elem)
        dice = (i*2)/u
        print(f"Dice for class label {classes[elem]} is {dice:.4f}")
        Dices.append(dice)
    return Dices

if __name__ == '__main__':
    gt_folder = "C:/Users\lok20\OneDrive\_Master\MAIA-ERASMUS/3 Semester\MIRA\Lab2/registered_labels"
    gts = os.listdir(gt_folder)
    img_folder = "C:/Users\lok20\OneDrive\_Master\MAIA-ERASMUS/3 Semester\MIRA\Lab2/registered_images"
    imgs = os.listdir(img_folder)


    gt_list = []
    for gt in gts:
        arr, _ = load_nifty_as_np(os.path.join(gt_folder, gt))
        gt_list.append(arr)

    img_list = []
    affine = None
    first = True
    for img in imgs:
        if first:
            arr, affine = load_nifty_as_np(os.path.join(img_folder, img))
            first = False
        else:
            arr, _ = load_nifty_as_np(os.path.join(img_folder, img))
        img_list.append(arr)

    tm = models.TissueModel(norm='none') # builds an anatomical and probabilistic atlas from a list of registered gt images
    tm.fit(img_list, gt_list)
    tm.show()
    tm.save("custom_tm")



