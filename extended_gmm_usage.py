import gc
import os
import nibabel as nib
import models
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def report_dices(scores):
    classes = {3: 'grey matter', 2: 'white matter', 1: 'csf'}
    for i in range(len(scores)):
        print('Image: ', i+1)
        for k in range(len(scores[i])):
            print(f"Dice for class label {classes[k+1]} is {scores[i][k]:.4f}")

def mean_dice(t1):
    print("COMPUTING MEAN STD DICE")
    classes = {3: 'grey matter', 2: 'white matter', 1: 'csf'}
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
    labels = [1, 2, 3]
    classes = {3:'grey matter', 2:'white matter', 1:'csf'}
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

    gt_folder = "your-labels-path"
    img_folder = "your-images-path"
    msk_folder = "your-mask-path"
    atlas_folder = "your-path-to-registered-atlas"
    imgs = os.listdir(img_folder)



    tm = models.TissueModel('hist')
    tm.load('custom_tm')

    dices = []
    for img in imgs:
        print('\n###############', img, '###############')
        at = models.Atlas()
        at.load(os.path.join(atlas_folder, img.split('.')[0]))
        prior = models.TMACombination(at, tm)
        image, affine = load_nifty_as_np(os.path.join(img_folder, img))
        mask, _ = load_nifty_as_np(os.path.join(msk_folder, img.split('.')[0]+'_1C.nii'))
        gt, _ = load_nifty_as_np(os.path.join(gt_folder, img.split('.')[0]+'_3C.nii'))
        TMA = models.TMACombination(at, tm)
        gmm = models.GMM(3, init='prior', prior=tm)

        seg = gmm.fit_transform(image, mask.astype(bool), True, 2)


        #sc = models.Scorer(gt, np.zeros(1), seg) # if kmeans is used
        #seg, _, dc = sc.relabel()# kmeans returns randomly ordered labels, scorer finds the best label combination

        dices.append(dice(seg, gt))
        nib.save(nib.Nifti1Image(seg, affine), os.path.join("your-output-path", img))
    mean_dice(dices)

