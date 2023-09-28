import torch
import torch.nn as nn
import torchvision
from fls.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fls.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
from fls.metrics.FLS import FLS
from fls.metrics.KID import KID
from fls.metrics.FID import FID
from fls.metrics.PrecisionRecall import PrecisionRecall

@torch.inference_mode()
def fls_score(train_dataset, test_dataset, gen_data, dataset_constant=1.322,
              train_dataset_name="CIFAR10_train", test_dataset_name="CIFAR10_test"):

    # Save path determines where features are cached (useful for train/test sets)
    feature_extractor = DINOv2FeatureExtractor(save_path="data/features")

    # FLS needs 3 sets of samples: train, test and generated
    train_dataset.name = train_dataset_name
    test_dataset.name = test_dataset_name
    # test_dataset.name = test_dataset_name

    train_feat = feature_extractor.get_all_features(train_dataset)
    test_feat = feature_extractor.get_all_features(test_dataset)

    gen_feat = feature_extractor.get_gen_features_from_data(
        gen_data, size=10000)
    # gen_feat = feature_extractor.get_gen_features(gen_data, size=10000)

    # 1.322 is a dataset specific constant
    fls = FLS("", dataset_constant).compute_metric(train_feat, test_feat, gen_feat)
    return fls

@torch.inference_mode()
def kid_fid_precision_recall_score(
        train_dataset, test_dataset, gen_data, dataset_constant=1.322,
        train_dataset_name="CIFAR10_train", test_dataset_name="CIFAR10_test"):

    # Save path determines where features are cached (useful for train/test sets)
    feature_extractor = InceptionFeatureExtractor(save_path="data/features")

    # FLS needs 3 sets of samples: train, test and generated
    train_dataset.name = train_dataset_name
    test_dataset.name = test_dataset_name

    train_feat = feature_extractor.get_all_features(train_dataset)
    test_feat = feature_extractor.get_all_features(test_dataset)

    # gen_feat = feature_extractor.get_gen_features(gen_data, size=10000)
    gen_feat = feature_extractor.get_gen_features_from_tensor(
        gen_data, batchsize=1024)
    # gen_feat = feature_extractor.get_gen_features_from_data(
    #     gen_data, size=10000)

    kid = KID().compute_metric(train_feat, test_feat, gen_feat)
    fid = FID().compute_metric(train_feat, test_feat, gen_feat)
    precision = PrecisionRecall(mode='Precision').compute_metric(
        train_feat, None, gen_feat)
    # recall = PrecisionRecall(mode='Recall').compute_metric(
    #     train_feat, None, gen_feat)
    recall = 0
    return kid, fid, precision, recall

@torch.inference_mode()
def fid_score(train_dataset, test_dataset, gen_data, dataset_constant=1.322,
              train_dataset_name="CIFAR10_train", test_dataset_name="CIFAR10_test"):

    # Save path determines where features are cached (useful for train/test sets)
    feature_extractor = InceptionFeatureExtractor(save_path="data/features")

    # FLS needs 3 sets of samples: train, test and generated
    train_dataset.name = train_dataset_name
    test_dataset.name = test_dataset_name

    train_feat = feature_extractor.get_all_features(train_dataset)
    test_feat = feature_extractor.get_all_features(test_dataset)

    gen_feat = feature_extractor.get_gen_features(gen_data, size=10000)

    fid = FID("",).compute_metric(train_feat, test_feat, gen_feat)
    return fid

@torch.inference_mode()
def precision_recall_score(train_dataset, test_dataset, gen_data,
                           mode='Precision', dataset_constant=1.322,
                           train_dataset_name="CIFAR10_train", test_dataset_name=None):

    # Save path determines where features are cached (useful for train/test sets)
    feature_extractor = InceptionFeatureExtractor(save_path="data/features")

    # FLS needs 3 sets of samples: train, test and generated
    train_dataset.name = train_dataset_name

    train_feat = feature_extractor.get_all_features(train_dataset)

    gen_feat = feature_extractor.get_gen_features(gen_data, size=10000)

    metric = PrecisionRecall("", mode=mode).compute_metric(train_feat, None, gen_feat)
    return metric
