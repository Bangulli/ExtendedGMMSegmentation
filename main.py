import gc
import os
import nibabel as nib
import models
import numpy as np
import scipy
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
    labels = np.unique(segmentation)
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

    gt_folder = "C:/Users\lok20\OneDrive\_Master\MAIA-ERASMUS/3 Semester\MISA\Lab3/test-set/testing-labels-unpacked"
    img_folder = "C:/Users\lok20\OneDrive\_Master\MAIA-ERASMUS/3 Semester\MISA\Lab3/test-set/testing-images-unpacked"
    msk_folder = "C:/Users\lok20\OneDrive\_Master\MAIA-ERASMUS/3 Semester\MISA\Lab3/test-set/testing-mask-unpacked"
    atlas_folder = "C:/Users\lok20\OneDrive\_Master\MAIA-ERASMUS/3 Semester\MISA\Lab3\custom_atlas_registered"
    imgs = os.listdir(img_folder)





    dices = []
    for img in imgs:
        at = models.Atlas()
        at.load(os.path.join(atlas_folder, img.split('.')[0]))
        image, affine = load_nifty_as_np(os.path.join(img_folder, img))
        mask, _ = load_nifty_as_np(os.path.join(msk_folder, img.split('.')[0]+'_1C.nii'))
        gt, _ = load_nifty_as_np(os.path.join(gt_folder, img.split('.')[0]+'_3C.nii'))
        gmm = models.GMM(3, at, init='atlas', verbose=False)
        seg = gmm.fit_transform(image, mask.astype(bool), True, 0)
        dices.append(dice(seg, gt))
        nib.save(nib.Nifti1Image(seg, affine), os.path.join("C:/Users\lok20\OneDrive\_Master\MAIA-ERASMUS/3 Semester\MISA\Lab3/result_atlas_post", img))
    mean_dice(dices)

    '''
    "C:/Users\lok20\OneDrive\_Master\MAIA-ERASMUS/3 Semester\MISA\Lab3/results_tissue-model_post" for tm init and posterior
    "C:/Users\lok20\OneDrive\_Master\MAIA-ERASMUS/3 Semester\MISA\Lab3/result_tissue-model_inf" for tm init and influence
    "C:/Users\lok20\OneDrive\_Master\MAIA-ERASMUS/3 Semester\MISA\Lab3/result_tissue-model" for tm init
    
    "C:/Users\lok20\OneDrive\_Master\MAIA-ERASMUS/3 Semester\MISA\Lab3/result_kmeans" for kmeans init
    
    "C:/Users\lok20\OneDrive\_Master\MAIA-ERASMUS/3 Semester\MISA\Lab3/result_atlas" for atlas init
    "C:/Users\lok20\OneDrive\_Master\MAIA-ERASMUS/3 Semester\MISA\Lab3/result_atlas_post" for atlas init and posterior
    "C:/Users\lok20\OneDrive\_Master\MAIA-ERASMUS/3 Semester\MISA\Lab3/result_atlas_init" for atlas init and influence
    '''
