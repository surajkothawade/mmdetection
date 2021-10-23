# import modules
import os
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import math

# import mmcv functionalities
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel, collate, scatter
from mmdet.apis import single_gpu_test, train_detector, init_detector, inference_detector, show_result_pyplot
from mmdet.datasets import (build_dataloader, build_dataset, replace_ImageToTensor)
from mmdet.datasets.dataset_wrappers import (ConcatDataset, RepeatDataset, ClassBalancedDataset)
from mmcv.ops import RoIPool
from mmdet.core import get_classes, bbox2roi, bbox_mapping, merge_aug_bboxes, merge_aug_masks, multiclass_nms
from mmdet.datasets.pipelines import Compose

#---------------------------------------------------------------------------#
#----------- Custom function to load dataset and return class --------------#
#---------------- wise object and image level statistics -------------------#
#---------------------------------------------------------------------------#

def get_class_statistics(dataset, indices):
  class_objects = {}   # define empty dict to hold class wise ground truths
  for i in range(len(dataset.CLASSES)):
    class_objects[i] = list()

  for i in indices:
    img_data, index = dataset[i]
    gt_labels = img_data['gt_labels'].data.numpy()
    for label in gt_labels:
        class_objects[label].append(index)
    
  #------------ print statistics -------------#
  print("Class".ljust(10), "No. of objects".ljust(3), "No. of images")
  print("-"*40)
  for key, val in class_objects.items():
    print(dataset.CLASSES[key].ljust(15), str(len(val)).ljust(15), len(set(val)))
  return class_objects

#---------------------------------------------------------------------------#
#--------------- Custom function to create Class Imbalance -----------------#
#---------------------------------------------------------------------------#

def create_custom_dataset(fullSet, all_indices, rare_class_budget, unrare_class_budget, imbalanced_classes, all_classes):

  labelled_budget = {}
  labelled_indices, unlabelled_indices = list(), list()
  exhausted_rare_classes = set()

  # initialize budget for rare and unrare class from the split_config file
  for i in range(len(fullSet.CLASSES)):
    if i in imbalanced_classes:
      labelled_budget[i] = rare_class_budget
    else:
      labelled_budget[i] = unrare_class_budget

  # iterate through whole dataset to select images class wise
  for i in all_indices:
    img_data, index = fullSet[i]
    #print(img_data)
    gt_labels = img_data['gt_labels'].data.numpy()
    
    # skip image if it does not contain classes with budget left
    if exhausted_rare_classes & set(gt_labels) or not (all_classes & set(gt_labels)):
      continue
    
    # else add image to the labelled pool and decrease budget class wise
    for label, no_of_objects in Counter(gt_labels).items():
        labelled_budget[label] -= no_of_objects # decrease budget

        if label in all_classes and labelled_budget[label] <= 0: # budget exhausted
          #print(fullSet.CLASSES[label]," class exhausted...")
          all_classes.remove(label)
          if label in imbalanced_classes:     # if rare class
            #print("added to rare class list")
            exhausted_rare_classes.add(label) # add to exhausted list of rare_classes
    
    labelled_indices.append(index)  # add image to labelled pool
    if not len(all_classes):        # if budget exceeded for all the classes, stop & return dataset
      #print("\nall class budget exhausted...")
      break


  # remove labelled indices from the full list
  labelled_indices = np.asarray(labelled_indices)
  unlabelled_indices = np.setdiff1d(all_indices, labelled_indices)

  # print dataset statistics
  stats = get_class_statistics(fullSet, labelled_indices)
  
  return labelled_indices, unlabelled_indices

#---------------------------------------------------------------------------#
#------------ Custom function to extract proposals from images -------------#
#---------------------------------------------------------------------------#

def extract_proposal_features(model, features, img_metas):
    assert model.with_bbox, 'Bbox head must be implemented.'

    proposal_list = model.rpn_head.simple_test_rpn(features, img_metas)
    return proposal_list

#---------------------------------------------------------------------------#
#---------------- Custom function to extract resized features --------------#
#------------------ from different layers after RoI Pooling ----------------#
#---------------------------------------------------------------------------#

def get_RoI_features(model, features, proposals, with_shared_fcs=False, only_cls_scores=False):
  """ Extract features from either the RoI pooling layers or shared Fully-connected layers
      or directly return the class_scores from the final class predictor itself.
    Args:
        model (nn.Module): The loaded detector.
        features  (tuple[Tensor]): Features from the upstream network, each is a 4D-tensor. 
        proposals (list[Tensor]): Either predicted proposals from RPN layers (unlabelled images)
                                  or transformed proposals from ground truth bounding boxes (query set).
        with_shared_fcs (Bool): if True, return the features from shared FC layer; default is False
        only_cls_scores (Bool): if True, return the class_scores from the final predictor; default is False
    Returns:
        (List[Tensor]) : If 'only_cls_scores' flag is set, class_scores from the final predictor for each 
                         proposal will be returned, otherwise return the feature maps after flattening out.
  """

  device = next(model.parameters()).device      # model device
  rois = bbox2roi(proposals).to(device=device)  # convert proposals to Region of Interests
  bbox_feats = model.roi_head.bbox_roi_extractor(
            features[:model.roi_head.bbox_roi_extractor.num_inputs], rois)
  
  if model.roi_head.with_shared_head:
            bbox_feats = model.roi_head.shared_head(bbox_feats)
  
  #print("Features shape from RoI Pooling Layer: ",bbox_feats.shape) # [no_of_proposals, 256, 7, 7]
  x = bbox_feats.flatten(1)       # flatten the RoI Pooling features

  if with_shared_fcs or only_cls_scores: # extract flattened features from shared FC layers
      for fc in model.roi_head.bbox_head.shared_fcs:
          x = model.roi_head.bbox_head.relu(fc(x))

      if only_cls_scores:           # if cls_scores flag is set
        cls_scores = model.roi_head.bbox_head.fc_cls(x) if model.roi_head.bbox_head.with_cls else None
        return cls_scores           # return class scores from the final class predictors
      # else return output from the shared_fc layers
      return x
  # else return features from the RoI pooling layer
  return bbox_feats

#---------------------------------------------------------------------------#
#-------- Define custom Uncertainty Score function : Score each image ------#
#-------- on the basis of Entropy and select the images with topmost -------#
#-------- Uncertainty scores for Next round of Uncertainty Sampling --------#
#---------------------------------------------------------------------------#

def get_uncertainty_scores(model, img_loader, no_of_imgs, imb_classes=None):

  uncertainty_scores = torch.zeros(no_of_imgs)
  device = next(model.parameters()).device  # model device

  if imb_classes is not None:
    print('using imbalanced classes ', imb_classes)

  for i, data_batch in enumerate(tqdm(img_loader)):     # for each batch
            
      # split the dataloader output into image_data and dataset indices
      img_data, indices = data_batch[0], data_batch[1].numpy()
      
      imgs, img_metas = img_data['img'].data[0].to(device=device), img_data['img_metas'].data[0]
      
      # extract image features from backbone + FPN neck
      with torch.no_grad():
          features = model.extract_feat(imgs)
      
      # get batch proposals from RPN Head and extract class scores from RoI Head
      batch_proposals = extract_proposal_features(model, features, img_metas)
      batch_cls_scores = get_RoI_features(model, features, batch_proposals, only_cls_scores=True)
      
      # normalize class_scores for each image to range between (0,1) which indicates
      # probability whether an object of that class has a bounding box centered there
      batch_cls_scores = batch_cls_scores.softmax(-1)

      # calculate class_entropies from the class probabilities
      # formula : entropy(p) = -[(p * logp) + {(1-p) * log(1-p)}] => (-p * logp) + {p * log(1-p)} - log(1-p)
      logp = torch.log2(batch_cls_scores)
      negp = torch.neg(batch_cls_scores)
      logOneMinusP = torch.log2(torch.add(negp, 1))
      batch_cls_scores = torch.add((negp * logp), torch.sub((batch_cls_scores * logOneMinusP),logOneMinusP))
      
      # split class_entropies as per no. of proposals in each image within batch
      num_proposals_per_img = tuple(len(p) for p in batch_proposals)
      batch_cls_scores = batch_cls_scores.split(num_proposals_per_img, 0)

      # for each image, take the max of class_entropies per proposal and aggregate over all proposals (average-max)
      for j, img_cls_scores in enumerate(batch_cls_scores):
        if imb_classes is not None:                 # use imbalanced class scores only for uncertainty score calculation
          imb_scores = torch.zeros(len(imb_classes))
          for k, imb_cls in enumerate(imb_classes):
            imb_scores[k] = torch.mean(img_cls_scores[:, imb_cls]) # average of each imb class over all proposals
          
          final_score = torch.max(imb_scores)                      # take max over all imb class averages
        else:                                       # use all class scores for uncertainty score calculation
          max_scores_per_proposal, _ = torch.max(img_cls_scores, dim=1) # take max of all class scores per proposal
          final_score = torch.mean(max_scores_per_proposal,dim=0)       # average over all proposals (avg-max implement)
        # store final uncertainty score for current image
        uncertainty_scores[indices[j]] = round(final_score.item(), 4)
      
  return uncertainty_scores

#---------------------------------------------------------------------------#
#------ Custom function to extract RoI features from Unlabelled set --------#
#---------------------------------------------------------------------------#

def get_unlabelled_RoI_features(model, unlabelled_loader, feature_type):

  device = next(model.parameters()).device  # model device
  unlabelled_indices = list()
  unlabeled_features = []
  if(feature_type == "fc"):
    fc_features = True
  for i, data_batch in enumerate(tqdm(unlabelled_loader)):     # for each batch
            
      # split the dataloader output into image_data and dataset indices
      img_data, indices = data_batch[0], data_batch[1].numpy()
      
      imgs, img_metas = img_data['img'].data[0].to(device=device), img_data['img_metas'].data[0]
      
      # extract image features from backbone + FPN neck
      with torch.no_grad():
        features = model.extract_feat(imgs)
      
      # get batch proposals from RPN Head and extract class scores from RoI Head
      batch_proposals = extract_proposal_features(model, features, img_metas)
      batch_roi_features = get_RoI_features(model, features, batch_proposals, with_shared_fcs=fc_features)
      
      num_proposals_per_img = tuple(len(p) for p in batch_proposals)
      batch_roi_features = batch_roi_features.split(num_proposals_per_img, 0)
            
      for j, img_roi_features in enumerate(batch_roi_features):
#         print(indices[j], img_roi_features.shape)
        unlabelled_indices.append(indices[j]) # add image index to list
        xf = img_roi_features.detach().cpu().numpy()
        unlabeled_features.append(xf)
#         xf = np.expand_dims(xf, axis=0)
#         if(len(unlabeled_features.shape)==1):
#           unlabeled_features = xf
#         else:
#           unlabeled_features = np.vstack((unlabeled_features, xf))
  unlabeled_features = np.stack(unlabeled_features, axis=0)
  return unlabeled_features, unlabelled_indices

#---------------------------------------------------------------------------#
#-------------- Custom function to Select Top-K Proposals ------------------#
#---------------------------------------------------------------------------#

def select_top_k_proposals(fg_cls_scores, fg_classes_with_max_score, fg_classes, proposal_budget):
  # get the indices in order which sorts the foreground class proposals scores in descending order
  max_score_order = torch.argsort(fg_cls_scores, descending=True).tolist()
  
  selected_prop_indices = list()
  # loop through until proposal budget is exhausted
  while proposal_budget:
    cls_budget, per_cls_budget, next_round_max_score_order =  dict(), (proposal_budget // len(fg_classes)) + 1, list()
    # assign budget to each foreground class
    for cls in fg_classes:
      cls_budget[cls.item()] = per_cls_budget
    
    # loop through the ordered list
    for idx in max_score_order:
      curr_class = fg_classes_with_max_score[idx].item()
      if cls_budget[curr_class]: # if budget permits
        selected_prop_indices.append(idx)   # add index to selection list
        cls_budget[curr_class] -= 1         # reduce class budget
        proposal_budget -= 1                # reduce proposal budget
        if not proposal_budget:             # stop if proposal budget exhausted
          break
      else:
        next_round_max_score_order.append(idx)
    # limit the order_list to indices not chosen in current iteration
    max_score_order = next_round_max_score_order
    
  return selected_prop_indices

#---------------------------------------------------------------------------#
#---------------- Custom function to extract RoI features ------------------#
#---------------- from Unlabelled set with Top-K Proposals -----------------#
#---------------------------------------------------------------------------#
def get_unlabelled_top_k_RoI_features(model, unlabelled_loader, proposal_budget, feature_type):

  device = next(model.parameters()).device  # model device
  unlabelled_indices = list()
  unlabelled_roi_features = list()

  if(feature_type == "fc"):
    fc_features = True

  for i, data_batch in enumerate(tqdm(unlabelled_loader)):     # for each batch
            
      # split the dataloader output into image_data and dataset indices
      img_data, img_indices = data_batch[0], data_batch[1].numpy()
      
      imgs, img_metas = img_data['img'].data[0].to(device=device), img_data['img_metas'].data[0]
      
      # extract image features from backbone + FPN neck
      with torch.no_grad():
          features = model.extract_feat(imgs)
      
      # get batch proposals from RPN Head and extract class scores from RoI Head
      batch_proposals = extract_proposal_features(model, features, img_metas)
      batch_roi_features = get_RoI_features(model, features, batch_proposals, with_shared_fcs=True)
      batch_cls_scores = get_RoI_features(model, features, batch_proposals, only_cls_scores=True)

      # normalize class_scores for each image to range between (0,1) which indicates
      # probability whether an object of that class has a bounding box centered there
      batch_cls_scores = batch_cls_scores.softmax(-1)
      
      # split features and cls_scores
      num_proposals_per_img = tuple(len(p) for p in batch_proposals)
      batch_cls_scores = batch_cls_scores.split(num_proposals_per_img, 0)
      batch_roi_features = batch_roi_features.split(num_proposals_per_img, 0)

      # for each image, select the top-k proposals where k = proposal_budget
      for j, img_cls_scores in enumerate(batch_cls_scores):
          img_roi_features = batch_roi_features[j]
          max_score_per_proposal, max_score_classes = torch.max(img_cls_scores, dim=1) # take max of all class scores per proposal
          classes, indices, counts = torch.unique(max_score_classes, return_inverse=True, return_counts=True)
          
          bg_class_index, bg_count, num_proposals  = len(classes) - 1, counts[-1], len(indices)
          fg_indices = indices != bg_class_index
          #print(classes, indices, counts)
          fg_img_cls_scores = max_score_per_proposal[fg_indices]
          fg_classes_with_max_score = max_score_classes[fg_indices]
          fg_img_roi_features = img_roi_features[fg_indices]
          #print(fg_img_roi_features.shape)
          
          if bg_count > num_proposals - proposal_budget: # no. of foreground proposals < proposal_budget
            #print("augment some background imgs")
            bg_indices = indices == bg_class_index
            bg_img_roi_features = img_roi_features[bg_indices][:bg_count - num_proposals + proposal_budget]
            selected_roi_features = torch.cat((fg_img_roi_features, bg_img_roi_features)).detach().cpu().numpy()
            del bg_indices, bg_img_roi_features
          elif bg_count == num_proposals - proposal_budget: # no. of foreground proposals = proposal_budget
            #print("no need to augment or select")
            selected_roi_features = fg_img_roi_features.detach().cpu().numpy()
          else:                                             # no. of foreground proposals > proposal_budget
            #print("select from foreground imgs")
            top_k_indices = select_top_k_proposals(fg_img_cls_scores, fg_classes_with_max_score, classes[:-1], proposal_budget)
            #print(fg_classes_with_max_score[top_k_indices])
            selected_roi_features = fg_img_roi_features[top_k_indices].detach().cpu().numpy()
          
          # append to unlebelled_roi_features list
          unlabelled_roi_features.append(selected_roi_features)
          unlabelled_indices.append(img_indices[j]) # add image index to list
          # free up gpu_memory
          del max_score_per_proposal, max_score_classes, classes, indices, counts, bg_class_index, bg_count, num_proposals,fg_indices, fg_img_cls_scores, fg_classes_with_max_score, fg_img_roi_features
          
  unlabelled_features = np.stack(unlabelled_roi_features, axis=0)
  return unlabelled_features, unlabelled_indices

#---------------------------------------------------------------------------#
#--------- Custom function to extract RoI features from Query set ----------#
#---------------------------------------------------------------------------#

def get_query_RoI_features(model, query_loader, imbalanced_classes, feature_type):

  device = next(model.parameters()).device  # model device
  query_indices = list()
  query_features = []
  if(feature_type == "fc"):
    fc_features = True
  for i, data_batch in enumerate(tqdm(query_loader)):     # for each batch
            
      # split the dataloader output into image_data and dataset indices
      img_data, indices = data_batch[0], data_batch[1].numpy()
      
      imgs, img_metas = img_data['img'].data[0].to(device=device), img_data['img_metas'].data[0]
      batch_gt_bboxes = img_data['gt_bboxes'].data[0]          # extract gt_bboxes from data batch
      batch_gt_labels = img_data['gt_labels'].data[0]          # extract gt_labels from data batch

      gt_bboxes, gt_labels = list(), list()
      # filter only the imbalanced class bboxes and labels
      for img_gt_bboxes, img_gt_labels in zip(batch_gt_bboxes, batch_gt_labels):
        #print(img_gt_bboxes, img_gt_labels)
        imb_cls_indices = torch.zeros(len(img_gt_labels), dtype=torch.bool)
        for imb_class in imbalanced_classes:
          imb_cls_indices = (imb_cls_indices | torch.eq(img_gt_labels, imb_class))
        
        #print('rare class:',img_gt_labels[imb_cls_indices], img_gt_bboxes[imb_cls_indices])
        gt_bboxes.append(img_gt_bboxes[imb_cls_indices])
        gt_labels.append(img_gt_labels[imb_cls_indices])
      
      num_gts_per_img = tuple(len(p) for p in gt_bboxes) # store how many bboxes per img
      #print(num_gts_per_img)
      #print(gt_bboxes, gt_labels)
      
      gt_bboxes = torch.cat(gt_bboxes)                   # stack all bboxes across batch of imgs
      gt_labels = torch.cat(gt_labels)                   # stack all labels across batch of imgs
      #print(gt_bboxes, gt_labels)
      
      # append confidence score of 1.0 to each gt_bboxes
      batch_proposals = torch.cat((gt_bboxes, torch.ones(gt_bboxes.shape[0], 1)), 1)
      # return batch proposals to original shape as were in batch
      batch_proposals =  batch_proposals.split(num_gts_per_img, 0)
      
      # extract image features from backbone + FPN neck
      with torch.no_grad():
          features = model.extract_feat(imgs)
      
      batch_roi_features = get_RoI_features(model, features, batch_proposals, with_shared_fcs=fc_features)
      batch_roi_features = batch_roi_features.split(num_gts_per_img, 0)
      
      for j, img_roi_features in enumerate(batch_roi_features):
        #print(indices[j], img_roi_features.shape)
        query_indices.append(indices[j]) # add image index to list
        xf = img_roi_features.detach().cpu().numpy()
        query_features.append(xf)
      
#   query_features = np.stack(query_features, axis=0)
  return query_features, query_indices

#---------------------------------------------------------------------------#
#------- Custom function to prepare Validation set from labelled set -------#
#---------------------------------------------------------------------------#

def prepare_val_file(trn_dataset, indices, filename_07='trainval_07.txt', filename_12='trainval_12.txt', strat_dir='.'):
  trnval_07_file = open(os.path.join(strat_dir, filename_07), 'w')
  trnval_12_file = open(os.path.join(strat_dir,filename_12), 'w')
  for i, index in enumerate(indices):
    img_prefix = trn_dataset[index][0]['img_metas'].data['filename'].split('/')[2]
    img_name = trn_dataset[index][0]['img_metas'].data['filename'].split('/')[-1].split('.')[0]
    if img_prefix == 'VOC2007':
      trnval_07_file.write(img_name + '\n')
    else:
      trnval_12_file.write(img_name + '\n')
  trnval_07_file.close()
  trnval_12_file.close()
  if os.path.getsize(trnval_07_file.name) and os.path.getsize(trnval_12_file.name):
    return [trnval_07_file.name, trnval_12_file.name]
  elif os.path.getsize(trnval_07_file.name):
    return trnval_07_file.name
  else:
    return trnval_12_file.name

#---------------------------------------------------------------------------#
#----------- Custom function for Query-Query kernel computation ------------#
#---------------------------------------------------------------------------#

def compute_queryQuery_kernel(query_dataset_feat):
    query_query_sim = []
    for i in range(len(query_dataset_feat)):
        query_row_sim = []
        for j in range(len(query_dataset_feat)):
            query_feat_i = query_dataset_feat[i] #(num_proposals, num_features)
            query_feat_j = query_dataset_feat[j]
            query_feat_i = l2_normalize(query_feat_i) 
            query_feat_j = l2_normalize(query_feat_j)
            dotp = np.tensordot(query_feat_i, query_feat_j, axes=([1],[1])) #compute the dot product along the feature dimension, i.e between every GT bbox of rare class in the query image
            max_match_queryGt_queryGt = np.amax(dotp, axis=(0,1)) #get the max from (num_proposals in query i, num_proposals in query j)
            query_row_sim.append(max_match_queryGt_queryGt)
        query_query_sim.append(query_row_sim)
    query_query_sim = np.array(query_query_sim)
    print("final query image kernel shape: ", query_query_sim.shape)
    return query_query_sim

#---------------------------------------------------------------------------#
#----------- Custom function for Query-Image kernel computation ------------#
#---------------------------------------------------------------------------#

def compute_queryImage_kernel(query_dataset_feat, unlabeled_dataset_feat):
    query_image_sim = []
    unlabeled_feat_norm = l2_normalize(unlabeled_dataset_feat) #l2-normalize the unlabeled feature vector along the feature dimension (batch_size, num_proposals, num_features)
    for i in range(len(query_dataset_feat)):
        query_feat = np.expand_dims(query_dataset_feat[i], axis=0)
        query_feat_norm = l2_normalize(query_feat) #l2-normalize the query feature vector along the feature dimension
        #print(query_feat_norm.shape)
        #print(unlabeled_feat_norm.shape)
        dotp = np.tensordot(query_feat_norm, unlabeled_feat_norm, axes=([2],[2])) #compute the dot product along the feature dimension, i.e between every GT bbox of rare class in the query image with all proposals from all images in the unlabeled set
        #print(dotp.shape)
        max_match_queryGt_proposal = np.amax(dotp, axis=(1,3)) #find the gt-proposal pair with highest similarity score for each image
        query_image_sim.append(max_match_queryGt_proposal)
    query_image_sim = np.vstack(tuple(query_image_sim))
    print("final query image kernel shape: ", query_image_sim.shape)
    return query_image_sim

#---------------------------------------------------------------------------#
#---------- Custom function for Image-Image kernel computation -------------#
#---------------------------------------------------------------------------#

def compute_imageImage_kernel(unlabeled_dataset_feat, batch_size=100):
    image_image_sim = []
    unlabeled_feat_norm = l2_normalize(unlabeled_dataset_feat) #l2-normalize the unlabeled feature vector along the feature dimension
    #print(unlabeled_feat_norm.shape)
    unlabeled_data_size = unlabeled_feat_norm.shape[0]
    for i in range(math.ceil(unlabeled_data_size/batch_size)): #batch through the unlabeled dataset to compute the similarity matrix
        start_ind = i*batch_size
        end_ind = start_ind + batch_size
        if(end_ind > unlabeled_data_size):
            end_ind = unlabeled_data_size
        unlabeled_feat_batch = unlabeled_feat_norm[start_ind:end_ind,:,:]
        dotp = np.tensordot(unlabeled_feat_batch, unlabeled_feat_norm, axes=([2],[2])) #compute the dot product along the feature dimension, i.e between every proposal in an unlabeled image with all proposals from all images in the unlabeled set
        #print(dotp.shape)
        max_match_unlabeledProposal_proposal = np.amax(dotp, axis=(1,3)) #find the proposal-proposal pair with highest similarity score for each image
        #print(max_match_unlabeledProposal_proposal.shape)
        image_image_sim.append(max_match_unlabeledProposal_proposal)
    image_image_sim = np.vstack(tuple(image_image_sim))
    print(image_image_sim.shape)
    return image_image_sim

#---------------------------------------------------------------------------#
#------------ Build the training dataset from the Config file --------------#
#---------------------------------------------------------------------------#

def build_dataset_with_indices(RepeatDataset):      # function to build dataset from config file and return with indices

    def __getitem__(self, index):
        data = RepeatDataset.__getitem__(self, index)
        return data, index

    return type(RepeatDataset.__name__, (RepeatDataset,), {
        '__getitem__': __getitem__,
    })

#---------------------------------------------------------------------------#
#------ Custom function to pass Training images through Test Pipeline ------#
#---------------------------------------------------------------------------#

def test_pipeline_images(model, imgs):
    """ Extract Convolutional features from the model backbone and Feature Pyramid Network neck.
    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image file names or loaded images.
    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the results directly.
    """
    print(imgs)
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'
    
    imgs = data['img'][0] #length=1
    return imgs

#---------------------------------------------------------------------------#
#---------------------- Custom function to L2 Normalize --------------------#
#---------------------------------------------------------------------------#

def l2_normalize(a, axis=-1, order=2):
    #L2 normalization that works for any arbitary axes
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

#---------------------------------------------------------------------------#
#-------- Custom function to Print Subprocess Output to Console ------------#
#---------------------------------------------------------------------------#
def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
#---------------------------------------------------------------------------#
#-------- Custom function to create Class Imbalance for BDD dataset --------#
#---------------------------------------------------------------------------#
def create_custom_dataset_bdd(fullSet, all_indices, rare_class_budget, unrare_class_budget, imbalanced_classes, all_classes, attr_details, img_attribute_dict):

  labelled_budget = {}
  labelled_indices, unlabelled_indices = list(), list()
  exhausted_rare_classes = set()
  attr_class, attr_property, attr_value, attr_budget = attr_details

  # initialize budget for rare and unrare class from the split_config file
  for i in range(len(fullSet.CLASSES)):
    if i in imbalanced_classes:
      labelled_budget[i] = rare_class_budget
    else:
      labelled_budget[i] = unrare_class_budget
  
  # iterate through whole dataset to select images class wise
  for k,i in enumerate(all_indices):
    img_data, index = fullSet[i]
    gt_labels = img_data['gt_labels'].data.numpy()
    #break
    # skip image if it does not contain classes with budget left
    img_name = img_data['img_metas'].data['filename'].split('/')[-1]
    img_attr = img_attribute_dict[img_name][attr_property]
    if attr_class in gt_labels and img_attr == attr_value:
      if attr_budget > 0:
        # print("attr budget = ", attr_budget)
        labelled_indices.append(index)
        attr_budget -= sum(gt_labels == attr_class)
      continue
    if exhausted_rare_classes & set(gt_labels) or not (all_classes & set(gt_labels)):
      continue
    
    # else add image to the labelled pool and decrease budget class wise
    for label, no_of_objects in Counter(gt_labels).items():
        labelled_budget[label] -= no_of_objects # decrease budget

        if label in all_classes and labelled_budget[label] <= 0: # budget exhausted
          # print(fullSet.CLASSES[label]," class exhausted...")
          all_classes.remove(label)
          if label in imbalanced_classes:     # if rare class
            # print("added to rare class list")
            exhausted_rare_classes.add(label) # add to exhausted list of rare_classes
    
    labelled_indices.append(index)  # add image to labelled pool
    if not len(all_classes):        # if budget exceeded for all the classes, stop & return dataset
      #print("\nall class budget exhausted...")
      break
  
  # remove labelled indices from the full list
  labelled_indices = np.asarray(labelled_indices)
  unlabelled_indices = np.setdiff1d(all_indices, labelled_indices)

  # print dataset statistics
  stats = get_class_statistics(fullSet, labelled_indices)
  
  return labelled_indices, unlabelled_indices

#---------------------------------------------------------------------------#
#---- Custom function to extract image wise attributes for BDD dataset -----#
#---------------------------------------------------------------------------#
def get_image_wise_attributes(json_file):
  # read det_train json file for image-attribute mapping
  rd_fl = open(json_file, 'r')
  str_data = rd_fl.read()
  image_data = json.loads(str_data)

  attribute_dict = {'weather': {'rainy': 0, 'snowy':0, 'clear':0, 'overcast':0, 'undefined':0, 'partly cloudy':0, 'foggy':0}, \
                    'scene': {'tunnel':0, 'residential':0, 'parking lot':0, 'undefined':0, 'city street':0, 'gas stations':0, 'highway':0}, \
                    'timeofday': {'daytime':0, 'night':0, 'dawn/dusk':0, 'undefined':0}}
  img_attribute_dict = {}
  img_names = list()
  for k,item in enumerate(image_data):
    w, d, s = item['attributes']['weather'], item['attributes']['timeofday'], item['attributes']['scene']
    attribute_dict['weather'][w] += 1
    attribute_dict['timeofday'][d] += 1
    attribute_dict['scene'][s] += 1
    img_attribute_dict[item['name']] = {'weather': item['attributes']['weather'], \
                                        'timeofday': item['attributes']['timeofday'], \
                                        'scene': item['attributes']['scene']}
  return attribute_dict, img_attribute_dict

def get_rare_attribute_statistics(dataset, indices, attr_details, img_attribute_dict):  
  selected_rare_indices, no_of_rare_obj = list(), 0   # define empty list to hold image indices with rare attributes
  attr_class, attr_property, attr_value, attr_budget = attr_details
  #print(len(indices))
  for i in indices:
    img_data, index = dataset[i]
    gt_labels = img_data['gt_labels'].data.numpy()
    img_name =  img_data['img_metas'].data['filename'].split('/')[-1]
    if attr_class in gt_labels and img_attribute_dict[img_name][attr_property] == attr_value:
        for label in gt_labels:
          if label == attr_class:
            no_of_rare_obj += 1
        selected_rare_indices.append(index)
  return selected_rare_indices, no_of_rare_obj