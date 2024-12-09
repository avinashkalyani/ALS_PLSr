"""Group classification comparison for face, foot and hand region:
1) call the subjects in different lists : control and als
2) load the functional data and the masks for hand, face, foot
3) mask the functional data
4) use SRM to do group classification """


import os
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import runpy
import time
import csv
from os.path import join as pjoin
from scipy import stats
import numpy as np
from multiprocessing import Pool
import subprocess
import random
from brainiak.funcalign.rsrm import RSRM
import matplotlib.pyplot as plt
from brainiak.funcalign.srm import SRM
from nilearn.image import load_img
from nilearn.masking import apply_mask
from scipy import signal
from scipy.stats import zscore
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import NuSVC
from sklearn import svm
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import cross_val_score
#from termcolor import colored
import nibabel as nib
from nilearn import datasets, image
#from brainiak.funcalign.sssrm import SSSRM
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut


"""importing als """

with open('/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project/als/als_list.txt') as inf:
    reader = csv.reader(inf, delimiter="\t")
    als = list(zip(*reader))[0]

"""importing control """
with open('/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project/control/control_list.txt') as inf:
    reader = csv.reader(inf, delimiter="\t")
    control = list(zip(*reader))[0]

subjects = [control, als]
subject_groups = ['control','als']
# group labels
l1=list(np.repeat(0,(len(control))))
l2 = list(np.repeat(1,(len(als))))
labels = np.concatenate((l1,l2))
labels = list(labels)

"""maybe move this function to func_utils later and try to run the same 
code once I know that this function is working"""


def run_crossval_both_proj(arrs1_y, arrs2_y,
                           arrs1_o, arrs2_o,
                           k, vox,
                           niter=30, ):
    # prepare empty results array # shape (nlabels, nruns, nsubs)
    nruns = 2
    nsubs = len(arrs1_y)

    """easy to call arrays"""
    arrs1_yo = [arrs1_y, arrs1_o]
    arrs2_yo = [arrs2_y, arrs2_o]
    # proj_age_run_dat =[]
    for age_idx in range(2):
        run1_arrs = arrs1_yo[age_idx]
        run2_arrs = arrs2_yo[age_idx]

        proj_run_dat_1 = []
        # for testsub_idx_c1 in range(int(len(arrs1)//2)):
        # Calculate the midpoint of the array
        subjs = int(len(run1_arrs))

        for trainrun_idx in range(2):
            projected_data_1 = list(np.zeros(nsubs))
            for testsub_idx in range(subjs):  # iterate over runs
                # select run used for training and test and according digit indices
                training_arrs_1 = (run1_arrs, run2_arrs)[trainrun_idx]
                training_arrs_2 = (arrs1_yo[abs(age_idx - 1)], arrs2_yo[abs(age_idx - 1)])[trainrun_idx]
                test_arrs_1 = (run1_arrs, run2_arrs)[abs(trainrun_idx - 1)]
                # we need only one test_data
                # test_arrs_2 = ((arrs1_yo[abs(age_idx - 1)], arrs2_yo[abs(age_idx - 1)]))[abs(trainrun_idx - 1)]

                start = time.time()
                print('starting run %i subject %i and age %i' % (trainrun_idx, testsub_idx, age_idx))

                # testsub_idxs = [testsub_idx_c1, testsub_idx_c2]
                trainsubs_traindata_1 = [x for i, x in enumerate(training_arrs_1) if
                                         i != testsub_idx]  # select training data
                trainsubs_traindata_2 = [x for i, x in enumerate(training_arrs_2) if i != testsub_idx]
                testsubs_traindata_1 = training_arrs_1[testsub_idx]
                # transform for 1st group:
                srm_1 = RSRM(n_iter=niter, features=k)  # train srm on training subject's training data
                srm_1.fit((trainsubs_traindata_1))
                w_1, s_1 = srm_1.transform_subject((testsubs_traindata_1))  # estimate test subject's bases
                srm_1.w_.insert(testsub_idx, w_1)
                srm_1.s_.insert(testsub_idx, s_1)
                # transform for 2nd group:
                srm_2 = RSRM(n_iter=niter, features=k)  # train srm on training subject's training data
                srm_2.fit((trainsubs_traindata_2))
                w_2, s_2 = srm_2.transform_subject((testsubs_traindata_1))  # estimate test subject's bases
                srm_2.w_.insert(testsub_idx, w_2)
                srm_2.s_.insert(testsub_idx, s_2)

                # transforming the testsubject's test_data #whole testarray
                shared_test_1, ind_terms_1 = srm_1.transform(test_arrs_1)  # project test run into shared space
                shared_test_2, ind_terms_2 = srm_2.transform(test_arrs_1)  # project test run into shared space
                projected_data_1[testsub_idx] = np.concatenate((shared_test_1[testsub_idx], shared_test_2[testsub_idx]),
                                                               axis=1)
                elapsed = time.time() - start
                print('this round took: ', elapsed)
                print(len(projected_data_1), projected_data_1[testsub_idx].shape)
            proj_run_dat_1.append(projected_data_1)
        print(len(proj_run_dat_1))
        print(vox, age_idx, k)
        np.savez('/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project/results/group_projection_rSRM_permuted_cyclic_%s_%s_%s.npz' % (
        vox, age_idx, k), projected_data=proj_run_dat_1)

    return None


"""Classification function"""


def loo_classify_balanced_comb(g1, g2, labels):
    loo = LeaveOneOut()
    X1, X2 = [], []
    for i in range(len(g1)):
        x1 = g1[i].flatten()
        x2 = g2[i].flatten()
        X1.append(x1)
        X2.append(x2)
    print(len(X1), X1[0].shape)
    print(len(X2), X2[0].shape)
    all_r1 = np.concatenate((X1, X2), axis=0)
    # all_r1 = np.nan_to_num(stats.zscore(all_r, axis=0, ddof=1))
    all_score2 = []
    for train_idx, test_idx in loo.split(X1):
        X1_train, X1_test = np.array(X1)[train_idx.astype(int)], np.array(X1)[test_idx.astype(int)]
        y1_train, y1_test = np.array(l1)[train_idx.astype(int)], np.array(l1)[test_idx.astype(int)]
        all_score = []
        for train_idx2, test_idx2 in loo.split(X2):
            X2_train, X2_test = np.array(X2)[train_idx2.astype(int)], np.array(X2)[test_idx2.astype(int)]
            X_test = np.concatenate((X1_test, X2_test))
            X_train = np.concatenate((X1_train, X2_train))

            y2_train, y2_test = np.array(l2)[train_idx2.astype(int)], np.array(l2)[test_idx2.astype(int)]
            y_train = np.concatenate((y1_train, y2_train))
            y_test = np.concatenate((y1_test, y2_test))
            # classifier = NuSVC(nu=0.5, kernel='rbf', gamma='auto')
            classifier = svm.SVC(kernel='linear')
            classifier = classifier.fit(X_train, y_train)
            predicted_labels = classifier.predict(X_test)
            score = accuracy_score(y_test, predicted_labels)
            all_score.append(score)

        print('class',predicted_labels, 'true', y_test, score)
        all_score2.append(all_score)
    print(np.mean(all_score2))
    mean_score = np.mean(all_score2)
    stdv = np.std(all_score2)
    print('starting test scores')
    cv = StratifiedKFold(3, shuffle=True)
    cv_scr, per_score, pval = permutation_test_score(classifier, all_r1, labels, scoring="accuracy", cv=cv,
                                                    n_permutations=1000)
    return all_score2, stdv, cv_scr, per_score, pval



""" loading the path for the data sets"""


def datagrabber(dsdir='/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project',
                testsubs=False,
                sub_ids=True):
    """
    # grab file names for
    # filtered bold data and roi masks from roi_glm output
    """
    run1_data, rhand_mask, face_mask, rfoot_mask = [], [], [],[]
    lhand_mask, lfoot_mask = [],[]
    masks = []
    for group_idx,sub_group in enumerate(subjects):
        group = subject_groups[group_idx]
        run1_data_g = []
        for indx, sub_id in enumerate(sub_group):
            run1 = pjoin(dsdir,group, sub_id, 'functional/sadata.nii')

            run1_data_g.append(run1)

        run1_data.append(run1_data_g)
    return run1_data


"""load the dataset paths"""
run1_data = datagrabber()
"""OK NOT SO SIMPLE: step1: load the spmt file and 
step2: split it in right and left hemisphere
step3: use corresponding hemisphere thresholded mask"""
# it will get complicated so lets just not do the splitting and just try with whole brain threshold and see

# so i didnt need to resample for the periodic simulation, a different approach


def fsl_out(out):
    th=''.join(out)
    threshold = []
    for t in th.split():
        try :
            threshold.append(float(t))
        except ValueError:
            pass
    TH = threshold[0]
    return TH
def thresholded_HEMI_bin(th_voxels, sub_ids, stat, group, body_part ):
    all_sub_vox = []
    for sub_idx, sub_id in enumerate(sub_ids):
        ds_dir = '/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project'
        out_path = pjoin(ds_dir, group,sub_id, 'results')
        isExist = os.path.exists(out_path)
        if not isExist:
            os.makedirs(out_path)
        # brain =nib.load(F_data[sub_idx])

        inp = pjoin(ds_dir, group ,sub_id, 'sanalyses',stat)

        command = f'fslstats {inp} -V'

        # Use subprocess to run the command and capture its output
        voxel = subprocess.check_output(command, shell=True, text=True)
        #voxel = !fslstats $inp - V
        # voxel is in list format so we form string first then split then take the first set of values
        # print(voxel)
        s = fsl_out(voxel)

        # making a mask for the first significant 300 voxels
        prop = th_voxels / s
        per = 100 - 100 * prop
        #thresh = !fslstats $inp - a - P $per

        command = f'/usr/local/fsl/bin/fslstats {inp} -a -P {per}'

        # Use subprocess to run the command
        try:
            thresh = subprocess.check_output(command, shell=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running the command: {e}")

        # voxel is in list format so we form string first then split then take the first set of values
        TH = fsl_out(thresh)
        bin_name = pjoin(out_path, 'VOXEL_MASK_bin_%s_%s.nii.gz' %(th_voxels,body_part))
        #!fslmaths $inp - thr $TH - bin $bin_name
        # Construct the command
        command = f'fslmaths {inp} -thr {TH} -bin {bin_name}'

        # Use subprocess to run the command
        try:
            subprocess.run(command, shell=True, check=True)
            print(f'fslmaths command executed successfully.')
        except subprocess.CalledProcessError as e:
            print(f'Error running the fslmaths command: {e}')
        all_sub_vox.append(s)
        print('all DONE! MASKS EXTRACTED for sub', sub_id, stat)
    return

the_voxels = [501]
#the_voxels = [200, 300, 500, 1000]
body_part = ['l_hand', 'r_hand', 'l_foot', 'r_foot', 'tongue']
spmT = ['spmT_0002.nii', 'spmT_0003.nii', 'spmT_0004.nii', 'spmT_0005.nii', 'spmT_0006.nii']

for stat_idx,stat in enumerate(spmT):
    for num_voxels in the_voxels:
        [thresholded_HEMI_bin(num_voxels,subjects[i],stat,group, body_part[stat_idx]) for i, group in enumerate(subject_groups)]

'''1500 looks good for now so lets do the analysis using that'''
# calling the body part masks
ds_dir = '/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project'
th_voxels = 1501
# Initialize a 3D list to store the bin_name paths
# Loop over body parts and groups
bin_name_list = []
for group_idx, group in enumerate(subject_groups):
    group_subjects = []  # Create a list for subjects in the current group
    for body_idx, part in enumerate(body_part):
        body_part_group = []  # Create a list for the current body part

        for sub_idx, sub_id in enumerate(subjects[group_idx]):
            # Construct the path to the mask directory
            mask_path = pjoin(ds_dir, group, sub_id, 'results')

            # Construct the binary mask file name based on the threshold and body part
            bin_name = pjoin(mask_path, f'VOXEL_MASK_bin_{th_voxels}_{part}.nii.gz')

            body_part_group.append(bin_name)

        group_subjects.append(body_part_group)

    bin_name_list.append(group_subjects)

"""apply mask and load arrays"""
def process_subject(group_idx):
    run_data = run1_data[group_idx]
    mask_image_parts = bin_name_list[group_idx]
    arrs_rhand = [np.nan_to_num(stats.zscore(apply_mask(load_img(data), mask_img=mask).T, axis=1, ddof=1))
                  for data, mask in zip(run_data, mask_image_parts[1])]
    arrs_lhand = [np.nan_to_num(stats.zscore(apply_mask(load_img(data), mask_img=mask).T, axis=1, ddof=1))
                  for data, mask in zip(run_data, mask_image_parts[0])]
    arrs_rfoot = [np.nan_to_num(stats.zscore(apply_mask(load_img(data), mask_img=mask).T, axis=1, ddof=1))
                  for data, mask in zip(run_data,  mask_image_parts[3])]
    arrs_lfoot = [np.nan_to_num(stats.zscore(apply_mask(load_img(data), mask_img=mask).T, axis=1, ddof=1))
                  for data, mask in zip(run_data,  mask_image_parts[2])]
    arrs_face = [np.nan_to_num(stats.zscore(apply_mask(load_img(data), mask_img=mask).T, axis=1, ddof=1))
                 for data, mask in zip(run_data,  mask_image_parts[4])]

    return arrs_rhand, arrs_lhand, arrs_rfoot, arrs_lfoot, arrs_face


# Number of threads to use
num_threads = 4

# Create a Pool of worker processes
pool = Pool(num_threads)

# Use the Pool to process subjects in parallel
result = pool.map(process_subject, range(len(subjects)))

# Close the Pool to release resources
pool.close()
pool.join()

# Unpack the results into separate lists
all_arrs_rhand, all_arrs_lhand, all_arrs_rfoot, all_arrs_lfoot, all_arrs_face = zip(*result)

#save the unpacked lists in a npz
# Define file names for saving
output_dir = "/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project/results"
npz_file_rhand = os.path.join(output_dir, "all_arrs_rhand.npz")
npz_file_lhand = os.path.join(output_dir, "all_arrs_lhand.npz")
npz_file_rfoot = os.path.join(output_dir, "all_arrs_rfoot.npz")
npz_file_lfoot = os.path.join(output_dir, "all_arrs_lfoot.npz")
npz_file_face = os.path.join(output_dir, "all_arrs_face.npz")

# Save the lists as separate .npz files
np.savez(npz_file_rhand, *all_arrs_rhand)
np.savez(npz_file_lhand, *all_arrs_lhand)
np.savez(npz_file_rfoot, *all_arrs_rfoot)
np.savez(npz_file_lfoot, *all_arrs_lfoot)
np.savez(npz_file_face, *all_arrs_face)

print("Saved the lists as .npz files.")

# Define the directory where the .npz files are located
input_dir = "/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project/results"

# Define the file paths for loading
npz_file_rhand = os.path.join(input_dir, "all_arrs_rhand.npz")
npz_file_lhand = os.path.join(input_dir, "all_arrs_lhand.npz")
npz_file_rfoot = os.path.join(input_dir, "all_arrs_rfoot.npz")
npz_file_lfoot = os.path.join(input_dir, "all_arrs_lfoot.npz")
npz_file_face = os.path.join(input_dir, "all_arrs_face.npz")

# Load the arrays from the .npz files with allow_pickle=True
data_rhand = np.load(npz_file_rhand, allow_pickle=True)
data_lhand = np.load(npz_file_lhand, allow_pickle=True)
data_rfoot = np.load(npz_file_rfoot, allow_pickle=True)
data_lfoot = np.load(npz_file_lfoot, allow_pickle=True)
data_face = np.load(npz_file_face, allow_pickle=True)

# Extract the arrays
all_arrs_rhand = [data_rhand[key] for key in data_rhand]
all_arrs_lhand = [data_lhand[key] for key in data_lhand]
all_arrs_rfoot = [data_rfoot[key] for key in data_rfoot]
all_arrs_lfoot = [data_lfoot[key] for key in data_lfoot]
all_arrs_face = [data_face[key] for key in data_face]

# Close the files
data_rhand.close()
data_lhand.close()
data_rfoot.close()
data_lfoot.close()
data_face.close()

def preprocess_array(arr, target_shape=(1500, 302)):
    """
    Preprocess a single array to have a specified target shape by zero-padding or cropping.

    Parameters:
    - arr: The array to preprocess.
    - target_shape: The desired shape for the array (default is (1500, 151)).

    Returns:
    - preprocessed_arr: The preprocessed array.
    """
    if arr.shape == target_shape:
        preprocessed_arr = arr
    else:
        pad_width = ((0, max(0, target_shape[0] - arr.shape[0])),
                     (0, max(0, target_shape[1] - arr.shape[1])))
        padded_arr = np.pad(arr, pad_width, mode='constant')
        preprocessed_arr = padded_arr[:target_shape[0], :target_shape[1]]

    return preprocessed_arr
def get_onsets_randomruns(subject_group,
                          ds_dir, group):
    # array with onsets of shape ngroup nparts, nsub
    onsets_array = np.zeros(shape=(12, 5, 4))
    for sub_idx, sub_id in enumerate(subject_group):
        for body_idx, body_int in enumerate(name_ons_body):
            dig_abspath = pjoin(ds_dir, group, sub_id, 'functional/time%s.ons' % body_int)
            with open(dig_abspath, 'r') as f:
                csv_reader = csv.reader(f, delimiter='\n')
                dig_onsets = [float(row[0]) for row in csv_reader]
                #print(sub_id, body_idx)
                onsets_array[sub_idx, body_idx] = dig_onsets
    return onsets_array

def randomruns_onsets_to_labels(randomruns_onsets_array,
                                stimdur=12,
                                tr=2.0,
                                nvols_in_random_run=302):
    stimdur_ms = int(stimdur * 100)
    nsubs, ndigits, nonsets_per_digit = randomruns_onsets_array.shape
    labels_ms = np.zeros(shape=(nsubs, int(nvols_in_random_run * tr * 100)))
    for sub_idx in range(nsubs):
        for digit_idx, digit_id in enumerate(range(1, ndigits + 1)):
            dig_onsets = randomruns_onsets_array[sub_idx, digit_idx]
            for ons in dig_onsets:
                ons_ms = int(ons * 100)
                labels_ms[sub_idx, int(ons_ms):int(ons_ms + stimdur_ms)] = digit_id
    random_labels = labels_ms[:, ::int(100 * tr)]
    return random_labels


def arrange_random(sub_ids, onsets_array, run_arrs):
    #onsets_array = get_onsets_randomruns(sub_ids, ds_dir)
    cyclic_arrs = list(np.zeros((len(sub_ids))))
    for sub_idx, sub_id in enumerate(sub_ids):
        data = onsets_array[sub_idx]  # FIRST SUBJ FIRST RUN
        d1_d5_conditions = ['D_%i' % i for i in range(1, 6)]
        df = pd.DataFrame(data.T, columns=d1_d5_conditions)
        df = df + 6
        df_TR_idx = np.ceil(df / 2 - 3)  # absolute index
        dur = 6
        all_dig = []
        print(run_arrs[sub_idx].shape)
        n_vox, nTRs = run_arrs[sub_idx].shape
        bin_trs = np.zeros((5, 4, dur, n_vox))  # soft code req
        for digit in enumerate(d1_d5_conditions):
            for d_idx in range(4):
                tr_idx = int(df_TR_idx.iloc[d_idx][digit[1]])
                data_ROI = run_arrs[sub_idx].T
                bin_trs[digit[0], d_idx, :, :] = data_ROI[tr_idx:tr_idx + dur, :]
        """attaching all the segments digit wise in cyclic order"""
        # joining the fingers
        stack1 = np.hstack((bin_trs))
        # joining the segments
        cyclic_rand = (stack1.reshape((5 * dur * 4), n_vox)).T

        cyclic_arrs[sub_idx] = cyclic_rand
    return cyclic_arrs
def arrange_random_labels(sub_ids, onsets_array, run_arrs):
    #onsets_array = get_onsets_randomruns(sub_ids, ds_dir)
    cyclic_arrs = list(np.zeros((len(sub_ids))))
    for sub_idx, sub_id in enumerate(sub_ids):
        data = onsets_array[sub_idx]  # FIRST SUBJ FIRST RUN
        d1_d5_conditions = ['D_%i' % i for i in range(1, 6)]
        df = pd.DataFrame(data.T, columns=d1_d5_conditions)
        df = df + 6
        df_TR_idx = np.ceil(df / 2 - 3)  # absolute index
        dur = 6
        all_dig = []
        nTRs = run_arrs[sub_idx].shape
        bin_trs = np.zeros((5, 4, dur))  # soft code req
        for digit in enumerate(d1_d5_conditions):
            for d_idx in range(4):
                tr_idx = int(df_TR_idx.iloc[d_idx][digit[1]])
                data_ROI = run_arrs[sub_idx].T
                bin_trs[digit[0], d_idx, :] = data_ROI[tr_idx:tr_idx + dur]
        """attaching all the segments digit wise in cyclic order"""
        # joining the fingers
        stack1 = np.hstack((bin_trs))
        # joining the segments
        cyclic_rand = (stack1.reshape((5 * dur * 4))).T

        cyclic_arrs[sub_idx] = cyclic_rand
    return cyclic_arrs
"""Classification function"""


ds_dir = "/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project"
body_part = ['l_hand', 'r_hand', 'l_foot', 'r_foot', 'tongue']

name_ons_body = ['HandLeft', 'HandRight','FootLeft', 'FootRight', 'Tongue'] # it was differnt before




"""NOTE1: the classes are unbalanced right now so maybe just call the first 11 subjects
in both the cases
NOTE2: since we dont have two runs, we split the data along the time dimension into two
151 each as part1 and part2"""

#arrays in body parts:
#all_arrs = [all_arrs_face, all_arrs_rhand, all_arrs_lhand, all_arrs_rfoot, all_arrs_lfoot]
all_arrs = [all_arrs_lhand, all_arrs_rhand, all_arrs_lfoot, all_arrs_rfoot, all_arrs_face]


k = 10
for body_idx, body_arr in enumerate(all_arrs):
    ons = get_onsets_randomruns(subjects[0], ds_dir, group='control')
    lab = randomruns_onsets_to_labels(ons)
    arrs = [preprocess_array(body_arr[0][i][:1500,:]) for i in range(11)]
    control_cyclic = arrange_random(subjects[0],ons,arrs)
    control_r1 = [control_cyclic[i][:,:60] for i in range(11)]
    control_r2 = [control_cyclic[i][:,60:302] for i in range(11)]

    ons = get_onsets_randomruns(subjects[1], ds_dir, group='als')
    lab = randomruns_onsets_to_labels(ons)
    arrs = [preprocess_array(body_arr[1][i][:1500, :]) for i in range(11)]
    als_cyclic = arrange_random(subjects[1],ons,arrs)
    als_r1 = [als_cyclic[i][:,:60] for i in range(11)]
    als_r2 = [als_cyclic[i][:,60:302] for i in range(11)]
    run_crossval_both_proj(control_r1, control_r2,
                           als_r1, als_r2,
                           k, body_idx,
                           niter=30, )





cv_scores = []
acc_scores_body = []
body_part = ['l_hand', 'r_hand', 'l_foot', 'r_foot', 'tongue']
for body_idx, v in enumerate(body_part):
    with np.load('/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project/results/group_projection_rSRM_permuted_cyclic_%s_%s_%s.npz'% (
    body_idx, 0, 10),
                 allow_pickle=True) as data:
        projected_data = data['projected_data']
    shared_all_y1 = projected_data[0]
    shared_all_y2 = projected_data[1]



    with np.load('/Volumes/IKND/AG_Kuehn/Avinash/ALS_Project/results/group_projection_rSRM_permuted_cyclic_%s_%s_%s.npz' % (
    body_idx, 1, 10),
                 allow_pickle=True) as data:
        projected_data = data['projected_data']
    shared_all_o1 = projected_data[0]
    shared_all_o2 = projected_data[1]
    a, b, c, d , e = loo_classify_balanced_comb(shared_all_y1, shared_all_o1, labels)
    print('done now next body part', a, b, c)
    scores_bod = [a, b, c, d, e]
    cv_scores.append(scores_bod)
    perm_score = d
    actual_score = c
    acc_scores_body.append(a)




"""---------------BAR PLOT----------------"""
import seaborn as sns
# Calculate mean and standard deviation for each body part
means = [np.mean(scores) for scores in acc_scores_body]
stds = [np.std(scores) for scores in acc_scores_body]
custom_colors = ['blue', 'red', 'cyan', 'yellow', 'green']

# Create bar plot
plt.figure(figsize=(6, 4))

# Plot bars with error bars
sns.barplot(x=body_part, y=means, ci="sd", palette=custom_colors)

# Add error bars on top of each bar
for i in range(len(body_part)):
    plt.errorbar(i, means[i], yerr=stds[i], fmt='o', color='black')

# Add text labels on top of each bar
for i in range(len(body_part)):
    plt.text(i, means[i]-0.3, f'{means[i]:.2f}', ha='center', va='bottom')

# Set plot title and labels
plt.title("Accuracy Scores for ALS vs Control across Different Body Parts")
plt.xlabel("Body Parts")
plt.ylabel("Accuracy Scores")
plt.tight_layout()
plt.savefig('figures/Group_prediction.png', dpi = 300)
# Show plot
plt.show()




"""---------------BAR PLOT END----------------"""
"""Permutation test results plot"""
fig, ax = plt.subplots()

actual_score = c #score_sub[0]
permuted_data = np.array(d)
# Create a figure and axes
fig, ax = plt.subplots()

# Plot the box plot of the permutation test distribution
ax.boxplot(permuted_data.T, showfliers=False)

# Plot the actual scores as red dots
ax.plot(range(1, 20), actual_score, 'ro')

# Set the labels and title
ax.set_xlabel('Subjects')
ax.set_ylabel('Scores')
ax.set_title('Permutation Test Distribution Tongue')
#plt.savefig('permutation_tests_random_young_Tongue.png', dpi=300)
