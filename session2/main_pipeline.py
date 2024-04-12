# -*- coding: utf-8 -*-
"""
morteza.zabihi@gmail.com

"""

import os
import time
import pickle
import scipy.io
from CHoRUS_Utilities import *
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

# =============================================================================   
# =============================================================================   
# =============================================================================

fs = 300 # sampling frequency
# Specify the folder path containing the .mat files
folder_path = 'training2017/'
# load the annotations
annotations = load_annotations(folder_path + "REFERENCE.csv")


def main():
    # =========================================================================  
    # ========================= Feature Extraction ============================  
    # =========================================================================   
    
    features = np.zeros((len(annotations), 97))
    labels = np.zeros((len(annotations),))
    start_time = time.time()
    
    for iter1 in range(len(annotations)):
        file = annotations[0].iloc[iter1] + ".mat"
        label = annotations[2].iloc[iter1]
        file_path = os.path.join(folder_path, file)
        data = scipy.io.loadmat(file_path)
        data = data['val']
        data = np.squeeze(data).astype(float)
        
        print("-"*50)
        print(f"[INFO:] {iter1+1}/{len(annotations)}, Loaded data: {file}, label: {annotations[1].iloc[iter1]}")
        # ----------------------------visualizations---------------------------
        # R_i2, _ = r_detection(data, fs, wd=5, plot=True, preprocess=True)
        # _, _, _ = spectrogram(data, fs, window_length=[], overlap=[], display=True)
        # _ = melspectrogram(data, fs, display=True, title='melspectrogram')    
        # ---------------------------------------------------------------------
        features[iter1, :] = feature_extraction(data, fs)
        labels[iter1] = label
    
    end_time = time.time()
    time_spent = end_time - start_time
    print(f"[INFO:] Time spent for feature extraction: , {time_spent/60:.4f} min")
    # -------------------------------------------------------------------------
    # save the features and labels
    with open('features.pkl', 'wb') as file:
        pickle.dump(features, file)
    
    with open('labels.pkl', 'wb') as file:
        pickle.dump(labels, file)
        
    print("[INFO:] Extracted features and labels are saved!")
    print("[INFO:] Number of extracted features: ", features.shape[1])
    # =========================================================================
    # ============================= Cross-validation ==========================
    # =========================================================================
    
    # load the features and labels
    with open('features.pkl', 'rb') as file:
        features = pickle.load(file)
    with open('labels.pkl', 'rb') as file:
        labels = pickle.load(file)
    # -------------------------------------------------------------------------
    
    # first let's do 5-fold cross-validation:
    n_folds = 5
    labels = labels.astype(int)
    # train_folds, test_folds = cross_validation(labels, n_folds=n_folds, random_state=42, method="kfold")
    train_folds, test_folds = cross_validation(labels, n_folds=n_folds, random_state=42, method="skfold")
    
    target_names = ['Normal', 'AF', 'Other', 'Noisy']
    metrics_list = []
    print("="*50)
    print("[INFO:] Cross-validation starts!")
    for iter1 in range(n_folds):
        
        print("-"*50)
        print(f"[INFO:] fold {iter1+1}/{n_folds}")
        # split into train and test -------------------------------------------
        train_indx = train_folds[iter1]
        test_indx = test_folds[iter1]
        
        features_train = features[train_indx, :]
        labels_train = labels[train_indx]
        
        features_test = features[test_indx, :]
        labels_test = labels[test_indx]
        # ---------------------------------------------------------------------
        # balance the classs-imbalance ----------------------------------------
        
        # train_data, train_labels, indbalanced =\
        #     balance_classes_random_downsampling(features_train, labels_train)
        
        train_data, train_labels, indbalanced =\
            balance_classes_random_upsampling(features_train, labels_train)
        # ---------------------------------------------------------------------
        # fit the classifier --------------------------------------------------
        model = CatBoostClassifier(iterations=1500,
                                   loss_function='MultiClass', 
                                   verbose=False,
                                   allow_writing_files=False)
        
        model.fit(train_data, train_labels)
        # ---------------------------------------------------------------------
        # get the feature importance ------------------------------------------
        
        # feature_importances = model.get_feature_importance()
        # ---------------------------------------------------------------------
        # Testing -------------------------------------------------------------
        
        predictions = model.predict_proba(features_test)
        # ---------------------------------------------------------------------
        # Evaluation ----------------------------------------------------------
        predicted_classes = np.argmax(predictions, axis=1)
        conf_matrix = confusion_matrix(labels_test, predicted_classes)  # Confusion Matrix
        print("[INFO:] Confusion Matrix:")
        print(conf_matrix)
        
        macro_f1 = f1_score(labels_test, predicted_classes, average='macro')
        micro_f1 = f1_score(labels_test, predicted_classes, average='micro')
        kl_score = kl_divergence_score(labels_test, predictions)
        auroc_ovr = roc_auc_score(labels_test, predictions, multi_class='ovr')
        auroc_ovo = roc_auc_score(labels_test, predictions, multi_class='ovo')
        print(f"[INFO:] KL Scores: {kl_score:.4f} - Micro F1: {micro_f1:.4f} - Macro F1: {macro_f1:.4f}")
        print(f"[INFO:] AUROC OVR: {auroc_ovr:.4f} - AUROC OVO: {auroc_ovo:.4f}")    
        
        # metrics of the physionet challenge 2017
        F1n = 2*conf_matrix[0,0] / (np.sum(conf_matrix[:,0]) + np.sum(conf_matrix[0,:]))
        F1a = 2*conf_matrix[1,1] / (np.sum(conf_matrix[:,1]) + np.sum(conf_matrix[1,:]))
        F1o = 2*conf_matrix[2,2] / (np.sum(conf_matrix[:,2]) + np.sum(conf_matrix[2,:]))
        F1p = 2*conf_matrix[3,3] / (np.sum(conf_matrix[:,3]) + np.sum(conf_matrix[3,:]))
        # F1 = (F1n + F1a + F1o + F1p ) / 4
        F1 = (F1n + F1a + F1o ) / 3
        print(f"[INFO:] Physionet score(F1): {F1:.4f} - F1n: {F1n:.4f} - F1a: {F1a:.4f} - F1o: {F1o:.4f} - F1p: {F1p:.4f}")
        
        # class_report = classification_report(labels_test, predicted_classes, target_names=target_names)
        # print("[INFO:] Classification Report:")
        # print(class_report)
        # ---------------------------------------------------------------------
        # Append metrics to the list
        metrics_list.append({
            'KL Score': kl_score,
            'Micro F1': micro_f1,
            'Macro F1': macro_f1,
            'AUROC OVR': auroc_ovr,
            'AUROC OVO': auroc_ovo,
            
            'F1n': F1n,
            'F1a': F1a,
            'F1o': F1o,
            'F1p': F1p,
            'F1': F1,
        })
        
    # Create DataFrame from the list
    metrics_df = pd.DataFrame(metrics_list)
    
    # Calculate mean and standard deviation
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()
    
    print("\nMean Metrics:")
    print(mean_metrics)
    print("\nStandard Deviation Metrics:")
    print(std_metrics)
    
if __name__ == "__main__":
    main()