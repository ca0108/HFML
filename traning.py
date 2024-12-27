import pandas as pd
import re
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score
import catboost as cb
from sklearn.cluster import KMeans

def preprocess_and_classify(ref_data_path, target_data_path):
    ref_data = pd.read_csv(ref_data_path)
    target_data = pd.read_csv(target_data_path)
    
    ref_formulas = ref_data.iloc[:, 0]
    target_formulas = target_data.iloc[:, 0]
    
    element_pattern = r'[A-Z][a-z]?'
    
    def extract_elements(formula):
        elements = re.findall(element_pattern, formula)
        return sorted(set(elements))
    
    ref_combinations = ref_formulas.apply(extract_elements)
    target_combinations = target_formulas.apply(extract_elements)
    
    def classify_combination(target, refs):
        target_set = set(target)
        for ref in refs:
            ref_set = set(ref)
            if len(target_set) >= len(ref_set) and ref_set.issubset(target_set):
                return 'PotentialHEO'
            elif len(target_set) <= len(ref_set) and target_set.issubset(ref_set):
                return 'PotentialHEO'
        return 'nonPotential'
    
    target_data['Classification'] = target_combinations.apply(lambda x: classify_combination(x, ref_combinations))
    
    return target_data

def clustering_and_stat_analysis(target_data):
    numeric_columns = target_data.select_dtypes(include=['number']).columns
    features = target_data[numeric_columns]

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=4, random_state=1)
    target_data['Cluster'] = kmeans.fit_predict(features_scaled)
    
    cluster_potential_counts = target_data[target_data['Classification'] == 'PotentialHEO'].groupby('Cluster').size()
    cluster_total_counts = target_data.groupby('Cluster').size()
    
    target_data.to_csv('classified_clustered_HEO.csv', index=False)
    cluster_stats = pd.DataFrame({
        'Cluster': cluster_total_counts.index,
        'Total_Counts': cluster_total_counts.values,
        'PotentialHEO_Counts': cluster_potential_counts.reindex(cluster_total_counts.index, fill_value=0).values
    })
    cluster_stats.to_csv('cluster_stats.csv', index=False)

    return cluster_stats, target_data

def catboost_and_roc_analysis(target_data, inputfile):
    df = pd.read_csv(inputfile, encoding='utf-8')
    df.drop(['XX'], axis=1, inplace=True)
    
    X = np.array(df.drop(['XX '], axis=1))
    Y = np.array(df['XX'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    classifier = cb.CatBoostClassifier(
        depth=10, iterations=100, l2_leaf_reg=1, learning_rate=0.01,
        eval_metric='AUC', verbose=0
    )
    
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    cv_auc = []
    plt.figure(figsize=(10, 6))
    
    os.makedirs("./ROC_Data", exist_ok=True)
    os.makedirs("./Confusion_Matrices", exist_ok=True)
    
    for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        print(f"Fitting fold {i+1}...")
        classifier.fit(X_train[train_index], y_train[train_index])
        y_val_pred = classifier.predict_proba(X_train[test_index])[:, 1]
        fpr, tpr, _ = roc_curve(y_train[test_index], y_val_pred)
        auc = roc_auc_score(y_train[test_index], y_val_pred)
        cv_auc.append(auc)
        plt.plot(fpr, tpr, lw=2, label=f'ROC fold {i+1} (area = {auc:.2f})')
        print(f"Fold {i+1} AUC: {auc:.2f}")
        
        roc_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        roc_data.to_csv(f'./ROC_Data/roc_fold_{i+1}.csv', index=False)
        
        y_fold_pred = classifier.predict(X_train[test_index])
        cm_fold = confusion_matrix(y_train[test_index], y_fold_pred)
        cm_df = pd.DataFrame(cm_fold, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
        cm_df.to_csv(f'./Confusion_Matrices/confusion_matrix_fold_{i+1}.csv')

    classifier.fit(X_train, y_train)
    
    model_path = 'your_model.cbm'
    loaded_classifier = cb.CatBoostClassifier()
    loaded_classifier.load_model(model_path)
    print("Model loaded from file")
    
    y_train_pred = loaded_classifier.predict_proba(X_train)[:, 1]
    y_test_pred = loaded_classifier.predict_proba(X_test)[:, 1]
    
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
    roc_auc_train = roc_auc_score(y_train, y_train_pred)
    plt.plot(fpr_train, tpr_train, color='blue', lw=2, label='ROC curve (training) (area = %0.2f)' % roc_auc_train)
    
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)
    roc_auc_test = roc_auc_score(y_test, y_test_pred)
    plt.plot(fpr_test, tpr_test, color='red', lw=2, label='ROC curve (testing) (area = %0.2f)' % roc_auc_test)
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()
    
    y_pred = loaded_classifier.predict(X_test)
    cm_test = confusion_matrix(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred)
    
    print("\nTesting F1 Score: {:.2f}".format(f1_test))
    
    cm_test_df = pd.DataFrame(cm_test, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    cm_test_df.to_csv('./Confusion_Matrices/confusion_matrix_test.csv')
    
    return cm_test_df, f1_test

def main():
    ref_data_path = 'your_ref_data_path'
    target_data_path = 'your_target_data_path'
    inputfile = 'your_input_data_path'
    
    target_data = preprocess_and_classify(ref_data_path, target_data_path)
    cluster_stats, clustered_data = clustering_and_stat_analysis(target_data)
    cm_test_df, f1_test = catboost_and_roc_analysis(clustered_data, inputfile)

    print(f"F1 Score for Testing: {f1_test}")
    print(f"Confusion Matrix for Testing: \n{cm_test_df}")

if __name__ == "__main__":
    main()
