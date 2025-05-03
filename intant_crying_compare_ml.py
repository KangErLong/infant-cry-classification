import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import tensorflow as tf
import xgboost as xgb
import gc
from tensorflow.keras import layers, models, Model, callbacks
import warnings
import argparse
warnings.filterwarnings('ignore')
import tensorflow_hub as hub
from tensorflow.keras import regularizers
from sklearn.metrics import roc_curve, auc, roc_auc_score


# 设置随机种子保证可复现性
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==================== 增强型数据处理模块 ====================
def extract_audio_features(y, sr):
    """提取增强的音频特征集，确保输出形状为32x32"""
    features = []
    
    # 基础时域特征 (4)
    features.append(np.mean(y))
    features.append(np.std(y))
    features.append(np.max(y) - np.min(y))
    features.append(np.mean(np.abs(np.diff(y))))
    
    # 频域特征 - MFCC (32维)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)
    features.extend(np.mean(mfcc, axis=1))
    
    # 频域特征 - 频谱质心 (1)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(cent))
    
    # 频域特征 - 频谱带宽 (1)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(bandwidth))
    
    # 频域特征 - 频谱滚降 (1)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))
    
    # 谐波与噪声特征 (1)
    try:
        hnr = librosa.effects.harmonic(y)
        features.append(np.mean(hnr[:min(len(hnr), len(y))]))
    except:
        features.append(0)
    
    # 小波变换特征 (32)
    try:
        cqt = np.abs(librosa.cqt(y, sr=sr))
        features.extend(np.mean(cqt, axis=1)[:32])
    except:
        features.extend([0] * 32)
    
    # 色度特征 (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    
    # 梅尔频谱 (32)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=32)
    features.extend(np.mean(mel, axis=1))
    
    # 零交叉率 (1)
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    
    # 频谱平坦度 (1)
    flatness = librosa.feature.spectral_flatness(y=y)
    features.append(np.mean(flatness))
    
    # 确保特征总数正好是1024 (32x32)
    if len(features) < 1024:
        features.extend([0] * (1024 - len(features)))
    elif len(features) > 1024:
        features = features[:1024]
    
    # 转换为numpy数组并reshape为32x32x1
    features_array = np.array(features).reshape(32, 32, 1)
    return features_array

def apply_audio_augmentation(y):
    """增强型音频数据增强"""
    augmented_samples = [('original', y)]
    # 时间拉伸/压缩
    try:
        y_stretch1 = librosa.effects.time_stretch(y, rate=0.8)
        y_stretch2 = librosa.effects.time_stretch(y, rate=1.2)
        augmented_samples.append(('stretch_0.8', y_stretch1))
        augmented_samples.append(('stretch_1.2', y_stretch2))
    except:
        pass
    # 添加不同级别的高斯噪声
    for noise_level in [0.005, 0.01]:
        noise = noise_level * np.random.randn(len(y))
        y_noise = y + noise
        augmented_samples.append((f'noise_{noise_level}', y_noise))
    # 音高偏移
    try:
        y_shifted = librosa.effects.pitch_shift(y, sr=16000, n_steps=2)
        augmented_samples.append(('pitch_shift', y_shifted))
    except:
        pass
    # 音量调整
    y_quiet = y * 0.5
    y_loud = np.clip(y * 1.5, -1, 1)
    augmented_samples.append(('volume_low', y_quiet))
    augmented_samples.append(('volume_high', y_loud))
    return augmented_samples

def load_and_process_data(data_path, max_len=3, sr=16000, test_size=0.2, force_reload=False):
    """增强型数据加载与处理，添加缓存机制避免重复处理"""
    # 检查是否存在缓存文件
    cache_dir = os.path.join(os.path.dirname(data_path), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"processed_data_{max_len}s_{sr}hz_{test_size}.npz")
    
    # 如果缓存文件存在且不强制重新加载，直接读取缓存
    if os.path.exists(cache_file) and not force_reload:
        print(f"Loading cached processed data from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        # 验证缓存数据的形状
        y_train = data['y_train']
        if len(y_train.shape) == 1:
            num_classes = len(np.unique(y_train))
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
            y_test = tf.keras.utils.to_categorical(data['y_test'], num_classes=num_classes)
        elif y_train.shape[1] == 1:
            num_classes = len(np.unique(y_train))
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
            y_test = tf.keras.utils.to_categorical(data['y_test'], num_classes=num_classes)
        else:
            y_train = data['y_train']
            y_test = data['y_test']
            
        X_train = data['X_train']
        X_test = data['X_test']
        print(f"Loaded data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
        return (X_train, X_test, 
                y_train, y_test)
    
    features = []
    labels = []
    processed_files = 0
    print(f"Loading data from {data_path}...")
    # 遍历数据目录
    class_counts = {}
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if not os.path.isdir(label_path):
            continue
        files = os.listdir(label_path)
        class_counts[label] = len(files)
        print(f"Found {len(files)} files in class '{label}'")
        for file in files:
            file_path = os.path.join(label_path, file)
            # 加载音频文件
            try:
                y, sr = librosa.load(file_path, sr=sr, duration=max_len)
                if len(y) < sr * 0.5:  # 跳过过短的音频
                    continue
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
            # 应用数据增强
            augmented_samples = apply_audio_augmentation(y)
            for aug_type, aug_y in augmented_samples:
                try:
                    # 提取增强特征集
                    feature_vector = extract_audio_features(aug_y, sr)
                    features.append(feature_vector)
                    labels.append(label)
                except Exception as e:
                    print(f"Error extracting features from {file_path} ({aug_type}): {e}")
            # 内存管理
            processed_files += 1
            if processed_files % 20 == 0:
                print(f"Processed {processed_files} files...")
                gc.collect()
    
    print(f"Total processed samples: {len(features)}")
    print(f"Class distribution: {class_counts}")
    # 转换为numpy数组并确保形状正确 (5875, 32, 32)
    X = np.array(features)
    # 确保所有特征数组都是32x32形状
    X = np.array([x.reshape(32, 32) if x.size == 1024 else np.pad(x, (0, 1024 - x.size)).reshape(32, 32) for x in X])
    # 验证输入形状并添加通道维度
    if X.ndim == 3:
        X = np.expand_dims(X, axis=-1)
    print(f'处理后数据形状: {X.shape}')
    print(f'数据类型验证: {X.dtype}, 最小值: {X.min()}, 最大值: {X.max()}')
    print(f'训练特征维度: {X.shape[1:]}')
    # 确保输入形状符合模型要求
    if X.shape[1:] != (32, 32, 1):
        X = X.reshape(-1, 32, 32, 1)
    elif X.shape[1:] == (32, 32):
        X = X.reshape(-1, 32, 32, 1)
    y = np.array(labels)
    # 确保输入形状为 (None, 32, 32, 1)
    if X.shape[1:] != (32, 32, 1):
        X = X.reshape(-1, 32, 32, 1)
    
    # 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # 转换为one-hot编码并确保形状正确
    num_classes = len(le.classes_)
    print(f'实际检测到的类别数量: {num_classes}')
    # 确保y_encoded是1D数组用于stratify
    if len(y_encoded.shape) > 1:
        y_encoded = np.argmax(y_encoded, axis=1)
    # 转换为one-hot编码
    y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)
    
    # 验证形状
    print(f'原始y_encoded形状: {y_encoded.shape}')
    print(f'原始y_onehot形状: {y_onehot.shape}')
    # 确保y_onehot是2D数组用于训练
    if len(y_onehot.shape) != 2 or y_onehot.shape[1] != num_classes:
        y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)
    # 验证类别数量
    print(f'Number of classes: {num_classes}')
    print(f'One-hot encoded labels shape: {y_onehot.shape}')
    
    # 确保训练和测试标签为one-hot编码的(样本数, num_classes)形状
    if y_onehot.shape[1] != num_classes:
        y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)
        print(f'Fixed one-hot encoding shape: {y_onehot.shape}')
    
    # 分割数据并保持one-hot编码
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, 
        test_size=test_size, 
        stratify=y_encoded,  # 使用原始编码标签进行分层
        random_state=SEED
    )
    
    # 保存处理后的数据到缓存
    np.savez_compressed(
        cache_file, 
        X_train=X_train, X_test=X_test, 
        y_train=y_train, y_test=y_test
    )
    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """训练和评估sklearn和keras模型"""
    # 获取类别数量
    num_classes = y_train.shape[1]
    # 转换标签格式为1D数组用于sklearn
    y_train_sk = np.argmax(y_train, axis=1)
    y_test_sk = np.argmax(y_test, axis=1)
    
    # 1. 训练和评估RandomForest模型
    print("\n=== RandomForest Classifier ===")
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    rf.fit(X_train.reshape(X_train.shape[0], -1), y_train_sk)
    
    # 训练集评估
    y_train_pred = rf.predict(X_train.reshape(X_train.shape[0], -1))
    print("\nTraining Set Metrics:")
    print(f"Accuracy: {accuracy_score(y_train_sk, y_train_pred):.4f}")
    print(f"Precision: {precision_score(y_train_sk, y_train_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_train_sk, y_train_pred, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_train_sk, y_train_pred, average='weighted'):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_train_sk, y_train_pred)}")
    
    # 测试集评估
    y_test_pred = rf.predict(X_test.reshape(X_test.shape[0], -1))
    print("\nTest Set Metrics:")
    print(f"Accuracy: {accuracy_score(y_test_sk, y_test_pred):.4f}")
    print(f"Precision: {precision_score(y_test_sk, y_test_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test_sk, y_test_pred, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_test_sk, y_test_pred, average='weighted'):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test_sk, y_test_pred)}")
    
    # 2. 训练和评估XGBoost模型
    print("\n=== XGBoost Classifier ===")
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        random_state=SEED,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    xgb_model.fit(X_train.reshape(X_train.shape[0], -1), np.argmax(y_train, axis=1))
    
    # 训练集评估
    y_train_pred = xgb_model.predict(X_train.reshape(X_train.shape[0], -1))
    print("\nTraining Set Metrics:")
    print(f"Accuracy: {accuracy_score(np.argmax(y_train, axis=1), y_train_pred):.4f}")
    print(f"Precision: {precision_score(np.argmax(y_train, axis=1), y_train_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(np.argmax(y_train, axis=1), y_train_pred, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(np.argmax(y_train, axis=1), y_train_pred, average='weighted'):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(np.argmax(y_train, axis=1), y_train_pred)}")
    
    # 测试集评估
    y_test_pred = xgb_model.predict(X_test.reshape(X_test.shape[0], -1))
    print("\nTest Set Metrics:")
    print(f"Accuracy: {accuracy_score(np.argmax(y_test, axis=1), y_test_pred):.4f}")
    print(f"Precision: {precision_score(np.argmax(y_test, axis=1), y_test_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(np.argmax(y_test, axis=1), y_test_pred, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(np.argmax(y_test, axis=1), y_test_pred, average='weighted'):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(np.argmax(y_test, axis=1), y_test_pred)}")
    
    # 3. 投票集成
    print("\n=== Voting Ensemble ===")
    from sklearn.ensemble import VotingClassifier
    
    estimators = [
        ('random_forest', rf),
        ('xgboost', xgb_model)
    ]
    
    voting = VotingClassifier(estimators=estimators, voting='soft')
    voting.fit(X_train.reshape(X_train.shape[0], -1), y_train_sk)
    
    # 训练集评估
    y_train_pred = voting.predict(X_train.reshape(X_train.shape[0], -1))
    print("\nTraining Set Metrics (Voting):")
    print(f"Accuracy: {accuracy_score(y_train_sk, y_train_pred):.4f}")
    print(f"Precision: {precision_score(y_train_sk, y_train_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_train_sk, y_train_pred, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_train_sk, y_train_pred, average='weighted'):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_train_sk, y_train_pred)}")
    
    # 测试集评估
    y_test_pred = voting.predict(X_test.reshape(X_test.shape[0], -1))
    print("\nTest Set Metrics (Voting):")
    print(f"Accuracy: {accuracy_score(y_test_sk, y_test_pred):.4f}")
    print(f"Precision: {precision_score(y_test_sk, y_test_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test_sk, y_test_pred, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_test_sk, y_test_pred, average='weighted'):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test_sk, y_test_pred)}")
    
    # 4. 堆叠集成
    print("\n=== Stacking Ensemble ===")
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    
    estimators = [
        ('random_forest', rf),
        ('xgboost', xgb_model),
        ('svm', SVC(kernel='rbf', probability=True, random_state=SEED)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ]
    
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=num_classes,
            random_state=SEED,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        ),
        cv=5
    )
    stacking.fit(X_train.reshape(X_train.shape[0], -1), y_train_sk)
    
    # 训练集评估
    y_train_pred = stacking.predict(X_train.reshape(X_train.shape[0], -1))
    print("\nTraining Set Metrics (Stacking):")
    print(f"Accuracy: {accuracy_score(y_train_sk, y_train_pred):.4f}")
    print(f"Precision: {precision_score(y_train_sk, y_train_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_train_sk, y_train_pred, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_train_sk, y_train_pred, average='weighted'):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_train_sk, y_train_pred)}")
    
    # 测试集评估
    y_test_pred = stacking.predict(X_test.reshape(X_test.shape[0], -1))
    print("\nTest Set Metrics (Stacking):")
    print(f"Accuracy: {accuracy_score(y_test_sk, y_test_pred):.4f}")
    print(f"Precision: {precision_score(y_test_sk, y_test_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test_sk, y_test_pred, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_test_sk, y_test_pred, average='weighted'):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test_sk, y_test_pred)}")
    
    return rf, xgb_model, voting, stacking

def plot_roc_curves(models, X_test, y_test_sk, model_names):
    """
    Plot ROC curves and calculate AUC for multiple models
    
    Args:
        models: List of trained models
        X_test: Test features
        y_test_sk: Test labels (non one-hot encoded)
        model_names: List of model names for the legend
    """
    plt.figure(figsize=(12, 8))
    
    # Get the number of classes
    n_classes = len(np.unique(y_test_sk))
    
    # For each model
    for model, model_name in zip(models, model_names):
        # Reshape data for sklearn models
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        
        # Get predicted probabilities
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test_reshaped)
        else:
            # For models without predict_proba
            y_score = np.zeros((X_test_reshaped.shape[0], n_classes))
            preds = model.predict(X_test_reshaped)
            for i, pred in enumerate(preds):
                y_score[i, pred] = 1
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # One-vs-Rest approach for multiclass
        for i in range(n_classes):
            # For each class, create binary labels (1 for current class, 0 for others)
            y_test_binary = (y_test_sk == i).astype(int)
            
            # Calculate FPR, TPR, and AUC
            fpr[i], tpr[i], _ = roc_curve(y_test_binary, y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        y_test_onehot = tf.keras.utils.to_categorical(y_test_sk, num_classes=n_classes)
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_onehot.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot micro-average ROC curve
        plt.plot(
            fpr["micro"], 
            tpr["micro"],
            label=f'{model_name} (AUC = {roc_auc["micro"]:.3f})',
            linewidth=2
        )
    
    # Plot random guessing line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    
    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('model_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print AUC scores for each class and each model
    print("\n=== AUC Scores by Class ===")
    for model, model_name in zip(models, model_names):
        print(f"\n{model_name}:")
        
        # Get predicted probabilities
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test_reshaped)
        else:
            y_score = np.zeros((X_test_reshaped.shape[0], n_classes))
            preds = model.predict(X_test_reshaped)
            for i, pred in enumerate(preds):
                y_score[i, pred] = 1
        
        # Calculate AUC for each class
        for i in range(n_classes):
            y_test_binary = (y_test_sk == i).astype(int)
            class_auc = roc_auc_score(y_test_binary, y_score[:, i])
            print(f"  Class {i}: {class_auc:.4f}")
        
        # Calculate micro and macro average AUC
        y_test_onehot = tf.keras.utils.to_categorical(y_test_sk, num_classes=n_classes)
        micro_auc = roc_auc_score(y_test_onehot, y_score, average='micro')
        macro_auc = roc_auc_score(y_test_onehot, y_score, average='macro')
        print(f"  Micro-average: {micro_auc:.4f}")
        print(f"  Macro-average: {macro_auc:.4f}")


if __name__ == "__main__":
    # 加载数据
    data_path = os.path.join(os.path.dirname(__file__), "infant_crying_data/train")
    X_train, X_test, y_train, y_test = load_and_process_data(data_path)
    
    # 训练和评估模型
    print("\n=== 开始模型训练和评估 ===")
    rf_model, xgb_model, voting_model, stacking_model = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # 获取非one-hot编码的测试标签
    y_test_sk = np.argmax(y_test, axis=1)
    
    # 绘制ROC曲线并计算AUC
    print("\n=== 绘制ROC曲线和计算AUC ===")
    models = [rf_model, xgb_model, voting_model, stacking_model]
    model_names = ['Random Forest', 'XGBoost', 'Voting Ensemble', 'Stacking Ensemble']
    plot_roc_curves(models, X_test, y_test_sk, model_names)
    
    print("\n=== 模型训练和评估完成 ===")


