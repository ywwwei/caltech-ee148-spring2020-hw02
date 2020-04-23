import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''

    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(box_1[0], box_2[0])
    y1 = max(box_1[1], box_2[1])
    x2 = min(box_1[2], box_2[2])
    y2 = min(box_1[3], box_2[3])

    # compute the area of intersection rectangle
    interArea = abs(max((x2 - x1, 0)) * max((y2 - y1), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    box1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
    box2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(box1_area + box2_area - interArea)

    # return the intersection over union value
    
    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.iteritems():
        gt = gts[pred_file]
        for i in range(len(gt)):
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])
                if iou > iou_thr and pred[j][-1] > conf_thr:
                    TP+=1
        FP = len(pred)-TP
        FN = len(gt)-TP

    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold. 

confidence_thrs = np.sort(np.array([preds_train[fname][4] for fname in preds_train],dtype=float)) # using (ascending) list of confidence scores as thresholds
IoU_thresholds = [0.25,0.5,0.75]
tp_train = np.zeros(len(confidence_thrs))
fp_train = np.zeros(len(confidence_thrs))
fn_train = np.zeros(len(confidence_thrs))
for IoU_threshold in IoU_thresholds:
    fig = plt.figure()
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=IoU_threshold, conf_thr=conf_thr)

    # Plot training set PR curves
    p_train = tp_train/(tp_train+fp_train)
    r_train = tp_train/(tp_train+fn_train)

    plt.plot(r_train, p_train, marker='.', label='IoU_thresholds='+str(IoU_thresholds))
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR curves for the training set')
plt.legend()
plt.show()

fig.savefig(os.path.join(preds_path,'PR_curves_train.png'),bbox_inches = 'tight')

if done_tweaking:
    print('Code for plotting test set PR curves.')

    confidence_thrs = np.sort(np.array([preds_test[fname][4] for fname in preds_test],dtype=float)) # using (ascending) list of confidence scores as thresholds
    IoU_thresholds = [0.25,0.5,0.75]
    tp_test = np.zeros(len(confidence_thrs))
    fp_test = np.zeros(len(confidence_thrs))
    fn_test = np.zeros(len(confidence_thrs))
    for IoU_threshold in IoU_thresholds:
        fig = plt.figure()
        for i, conf_thr in enumerate(confidence_thrs):
            tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=IoU_threshold, conf_thr=conf_thr)

        # Plot training set PR curves
        p_test = tp_test/(tp_test+fp_test)
        r_test = tp_test/(tp_test+fn_test)

        plt.plot(r_test, p_test, marker='.', label='IoU_thresholds='+str(IoU_thresholds))
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curves for the testing set')
    plt.legend()
    plt.show()

    fig.savefig(os.path.join(preds_path,'PR_curves_test.png'),bbox_inches = 'tight')
