import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


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

# ==================== 新增加的深度学习模型包装器 ====================
class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    """为keras模型创建scikit-learn兼容的包装器"""
    
    def __init__(self, model, reshape_func=None, classes_=None):
        self.model = model
        self.reshape_func = reshape_func
        self.classes_ = classes_
        # 确保所有必需的sklearn属性都存在
        self._estimator_type = "classifier"
        
    def fit(self, X, y):
        # 实际上这是一个已经训练好的模型包装器，不需要再训练
        if self.classes_ is None:
            self.classes_ = np.unique(np.argmax(y, axis=1) if len(y.shape) > 1 else y)
            self.classes = self.classes_  # Update both attributes
        return self
        
    def predict(self, X):
        # 应用reshape函数（如果需要）
        if self.reshape_func is not None:
            X = self.reshape_func(X)
        # 使用模型进行预测并获取类别索引
        return np.argmax(self.model.predict(X), axis=1)
    
    def predict_proba(self, X):
        # 应用reshape函数（如果需要）
        if self.reshape_func is not None:
            X = self.reshape_func(X)
        # 返回概率预测
        return self.model.predict(X)
        
    def get_params(self, deep=True):
        # 实现get_params方法以支持sklearn的克隆
        return {
            'model': self.model,
            'reshape_func': self.reshape_func,
            'classes_': self.classes_
        }
    
    # Add a method to support scikit-learn's clone operation
    def __sklearn_clone__(self):
        return KerasClassifierWrapper(
            model=self.model,
            reshape_func=self.reshape_func,
            classes_=self.classes_
        )

# ==================== 层次化集成模型实现 ====================
def build_hierarchical_ensemble(models_dict, X_train, X_test, y_train, y_test, X_train_lstm, X_test_lstm):
    """
    构建层次化集成模型
    
    第一层：基础模型分组
        - 传统机器学习组：Random Forest, XGBoost, SVM, KNN
        - 序列模型组：LSTM, GRU, Transformer
        - CNN模型
    
    第二层：组内集成
        - 传统ML组 -> 使用投票集成
        - 序列模型组 -> 使用stacking集成
    
    第三层：最终集成
        - 使用stacking将各组集成模型的结果合并
    """
    # 获取标签
    num_classes = y_train.shape[1]
    y_train_sk = np.argmax(y_train, axis=1)
    y_test_sk = np.argmax(y_test, axis=1)
    
    print("\n=== 构建层次化集成模型 ===")
    
    # 创建CNN模型包装器
    def reshape_to_cnn(X):
        if X.shape[1:] != (32, 32, 1):
            return X.reshape(-1, 32, 32, 1)
        return X
    
    cnn_wrapper = KerasClassifierWrapper(
        model=models_dict['cnn'],
        reshape_func=reshape_to_cnn,
        classes_=np.arange(num_classes)
    )
    
    # 创建LSTM模型包装器
    def reshape_to_lstm(X):
        if len(X.shape) == 4:  # 如果是(n, 32, 32, 1)的CNN输入
            return X.reshape(-1, 32, 32)
        elif X.shape[1:] == (1024,):  # 如果是展平的特征
            return X.reshape(-1, 32, 32)
        return X
    
    lstm_wrapper = KerasClassifierWrapper(
        model=models_dict['lstm'],
        reshape_func=reshape_to_lstm,
        classes_=np.arange(num_classes)
    )
    
    # 创建GRU模型包装器
    gru_wrapper = KerasClassifierWrapper(
        model=models_dict['gru'],
        reshape_func=reshape_to_lstm,
        classes_=np.arange(num_classes)
    )
    
    # 创建Transformer模型包装器
    transformer_wrapper = KerasClassifierWrapper(
        model=models_dict['transformer'],
        reshape_func=reshape_to_lstm,
        classes_=np.arange(num_classes)
    )
    
    # =================第一层：基础模型分组=================
    # 1.1 传统机器学习组
    print("\n--- 第一层：基础模型组 ---")
    print("1.1 传统机器学习组")
    
    # 创建SVM和KNN模型
    svm_model = SVC(probability=True, random_state=SEED)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    svm_model.fit(X_train_flat, y_train_sk)
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_flat, y_train_sk)
    
    # 评估单个传统ML模型
    trad_models = {
        'Random Forest': models_dict['random_forest'],
        'XGBoost': models_dict['xgboost'],
        'SVM': svm_model,
        'KNN': knn_model
    }
    
    for name, model in trad_models.items():
        y_pred = model.predict(X_test_flat)
        acc = accuracy_score(y_test_sk, y_pred)
        f1 = f1_score(y_test_sk, y_pred, average='weighted')
        print(f"{name} - 准确率: {acc:.4f}, F1分数: {f1:.4f}")
    
    # 1.2 序列模型组
    print("\n1.2 序列模型组")
    
    # 评估单个序列模型
    seq_models = {
        'LSTM': lstm_wrapper,
        'GRU': gru_wrapper,
        'Transformer': transformer_wrapper
    }
    
    for name, model in seq_models.items():
        y_pred = model.predict(X_test_lstm)
        acc = accuracy_score(y_test_sk, y_pred)
        f1 = f1_score(y_test_sk, y_pred, average='weighted')
        print(f"{name} - 准确率: {acc:.4f}, F1分数: {f1:.4f}")
    
    # 1.3 CNN模型
    print("\n1.3 CNN模型")
    
    cnn_pred = cnn_wrapper.predict(X_test)
    cnn_acc = accuracy_score(y_test_sk, cnn_pred)
    cnn_f1 = f1_score(y_test_sk, cnn_pred, average='weighted')
    print(f"CNN - 准确率: {cnn_acc:.4f}, F1分数: {cnn_f1:.4f}")
    
    # =================第二层：组内集成=================
    print("\n--- 第二层：组内集成 ---")
    
    # 2.1 传统机器学习组集成
    print("2.1 传统机器学习组集成")
    trad_estimators = [
        ('random_forest', models_dict['random_forest']),
        ('xgboost', models_dict['xgboost']),
        ('svm', svm_model),
        ('knn', knn_model)
    ]
    
    # 使用stacking集成
    trad_ensemble = StackingClassifier(
        estimators=trad_estimators,
        final_estimator=LogisticRegression(multi_class='multinomial', solver='lbfgs'),
        cv=5
    )
    trad_ensemble.fit(X_train_flat, y_train_sk)
    
    # 评估传统ML集成
    trad_ens_pred = trad_ensemble.predict(X_test_flat)
    trad_ens_acc = accuracy_score(y_test_sk, trad_ens_pred)
    trad_ens_f1 = f1_score(y_test_sk, trad_ens_pred, average='weighted')
    print(f"传统ML集成 - 准确率: {trad_ens_acc:.4f}, F1分数: {trad_ens_f1:.4f}")
    
    # 2.2 序列模型组集成
    print("\n2.2 序列模型组集成")
    seq_estimators = [
        ('lstm', lstm_wrapper),
        ('gru', gru_wrapper),
        ('transformer', transformer_wrapper)
    ]
    
    # 使用stacking集成
    seq_ensemble = StackingClassifier(
        estimators=seq_estimators,
        final_estimator=LogisticRegression(multi_class='multinomial', solver='lbfgs'),
        cv=5
    )
    
    # 为序列模型准备统一格式的数据
    X_train_seq = X_train.reshape(X_train.shape[0], 32, 32)
    X_test_seq = X_test.reshape(X_test.shape[0], 32, 32)
    
    seq_ensemble.fit(X_train_seq, y_train_sk)
    
    # 评估序列模型集成
    seq_ens_pred = seq_ensemble.predict(X_test_seq)
    seq_ens_acc = accuracy_score(y_test_sk, seq_ens_pred)
    seq_ens_f1 = f1_score(y_test_sk, seq_ens_pred, average='weighted')
    print(f"序列模型集成 - 准确率: {seq_ens_acc:.4f}, F1分数: {seq_ens_f1:.4f}")
    
    # =================第三层：最终集成=================
    print("\n--- 第三层：最终层次化集成 ---")
    
    # 为最终集成准备数据
    # 从第一层各模型获取训练集预测概率
    print("从各模型获取训练集预测...")
    
    # 传统ML模型预测
    trad_train_proba = np.zeros((X_train.shape[0], num_classes*len(trad_models)))
    col_idx = 0
    for model_name, model in trad_models.items():
        proba = model.predict_proba(X_train_flat)
        for class_idx in range(num_classes):
            trad_train_proba[:, col_idx] = proba[:, class_idx]
            col_idx += 1
    
    # 序列模型预测
    seq_train_proba = np.zeros((X_train.shape[0], num_classes*len(seq_models)))
    col_idx = 0
    for model_name, model in seq_models.items():
        proba = model.predict_proba(X_train_seq)
        for class_idx in range(num_classes):
            seq_train_proba[:, col_idx] = proba[:, class_idx]
            col_idx += 1
    
    # CNN模型预测
    cnn_train_proba = cnn_wrapper.predict_proba(X_train)
    
    # 组合所有模型组的预测作为最终集成的特征
    X_train_meta = np.hstack([
        trad_train_proba, 
        seq_train_proba, 
        cnn_train_proba
    ])
    
    # 同样为测试集准备元特征
    print("从各模型获取测试集预测...")
    
    # 传统ML模型预测
    trad_test_proba = np.zeros((X_test.shape[0], num_classes*len(trad_models)))
    col_idx = 0
    for model_name, model in trad_models.items():
        proba = model.predict_proba(X_test_flat)
        for class_idx in range(num_classes):
            trad_test_proba[:, col_idx] = proba[:, class_idx]
            col_idx += 1
    
    # 序列模型预测
    seq_test_proba = np.zeros((X_test.shape[0], num_classes*len(seq_models)))
    col_idx = 0
    for model_name, model in seq_models.items():
        proba = model.predict_proba(X_test_seq)
        for class_idx in range(num_classes):
            seq_test_proba[:, col_idx] = proba[:, class_idx]
            col_idx += 1
    
    # CNN模型预测
    cnn_test_proba = cnn_wrapper.predict_proba(X_test)
    
    # 组合所有模型组的预测作为最终集成的特征
    X_test_meta = np.hstack([
        trad_test_proba, 
        seq_test_proba, 
        cnn_test_proba
    ])
    
    # 计算类别权重
    class_counts = np.bincount(y_train_sk)
    class_weights = {i: sum(class_counts)/class_counts[i] for i in range(num_classes)}
    
    # 使用更复杂的StackingClassifier作为最终集成器
    final_ensemble = StackingClassifier(
        estimators=[
            ('xgb', xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=num_classes,
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=SEED,
                scale_pos_weight=class_weights
            )),
            ('lr', LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                class_weight='balanced',
                max_iter=1000
            )),
            ('rf', RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced_subsample',
                random_state=SEED
            ))
        ],
        final_estimator=LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            class_weight='balanced',
            max_iter=1000
        ),
        cv=5
    )
    
    # 训练最终集成器（带类别权重）
    print("训练层次化集成最终模型...")
    final_ensemble.fit(X_train_meta, y_train_sk)
    
    # 最终预测与评估
    final_pred = final_ensemble.predict(X_test_meta)
    final_acc = accuracy_score(y_test_sk, final_pred)
    final_f1 = f1_score(y_test_sk, final_pred, average='weighted')
    final_precision = precision_score(y_test_sk, final_pred, average='weighted')
    final_recall = recall_score(y_test_sk, final_pred, average='weighted')
    
    print("\n=== 层次化集成模型最终评估结果 ===")
    print(f"准确率: {final_acc:.4f}")
    print(f"精确率: {final_precision:.4f}")
    print(f"召回率: {final_recall:.4f}")
    print(f"F1分数: {final_f1:.4f}")
    print(f"混淆矩阵:\n{confusion_matrix(y_test_sk, final_pred)}")
    
    # 计算ROC曲线和AUC
    final_proba = final_ensemble.predict_proba(X_test_meta)
    final_auc_micro = roc_auc_score(tf.keras.utils.to_categorical(y_test_sk, num_classes), final_proba, average='micro')
    final_auc_macro = roc_auc_score(tf.keras.utils.to_categorical(y_test_sk, num_classes), final_proba, average='macro')
    
    print(f"微平均AUC: {final_auc_micro:.4f}")
    print(f"宏平均AUC: {final_auc_macro:.4f}")
    
    # 返回最终集成模型和所有中间集成
    hierarchical_ensemble = {
        'trad_ensemble': trad_ensemble,
        'seq_ensemble': seq_ensemble,
        'final_ensemble': final_ensemble,
        'x_train_meta': X_train_meta,
        'x_test_meta': X_test_meta
    }
    
    return hierarchical_ensemble

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
    
    # ==================== 新增深度学习模型 ====================
    # 5. CNN模型
    print("\n=== CNN Model ===")
    # 创建早停回调
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    # 创建CNN模型
    cnn_model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # 编译CNN模型
    cnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 训练CNN模型
    cnn_history = cnn_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 评估CNN模型
    cnn_train_loss, cnn_train_acc = cnn_model.evaluate(X_train, y_train, verbose=0)
    cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
    
    # 获取预测结果
    y_train_pred_cnn = np.argmax(cnn_model.predict(X_train), axis=1)
    y_test_pred_cnn = np.argmax(cnn_model.predict(X_test), axis=1)
    
    # 打印评估指标
    print("\nTraining Set Metrics (CNN):")
    print(f"Accuracy: {cnn_train_acc:.4f}")
    print(f"Precision: {precision_score(y_train_sk, y_train_pred_cnn, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_train_sk, y_train_pred_cnn, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_train_sk, y_train_pred_cnn, average='weighted'):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_train_sk, y_train_pred_cnn)}")
    
    print("\nTest Set Metrics (CNN):")
    print(f"Accuracy: {cnn_test_acc:.4f}")
    print(f"Precision: {precision_score(y_test_sk, y_test_pred_cnn, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test_sk, y_test_pred_cnn, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_test_sk, y_test_pred_cnn, average='weighted'):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test_sk, y_test_pred_cnn)}")
    
    # 6. LSTM模型
    print("\n=== LSTM Model ===")
    # 为LSTM准备数据 - 需要将2D特征转换为序列
    X_train_lstm = X_train.reshape(X_train.shape[0], 32, 32)
    X_test_lstm = X_test.reshape(X_test.shape[0], 32, 32)
    
    # 创建LSTM模型
    lstm_model = models.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=(32, 32)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.LSTM(64),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # 编译LSTM模型
    lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 训练LSTM模型
    lstm_history = lstm_model.fit(
        X_train_lstm, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 评估LSTM模型
    lstm_train_loss, lstm_train_acc = lstm_model.evaluate(X_train_lstm, y_train, verbose=0)
    lstm_test_loss, lstm_test_acc = lstm_model.evaluate(X_test_lstm, y_test, verbose=0)
    
    # 获取预测结果
    y_train_pred_lstm = np.argmax(lstm_model.predict(X_train_lstm), axis=1)
    y_test_pred_lstm = np.argmax(lstm_model.predict(X_test_lstm), axis=1)
    
    # 打印评估指标
    print("\nTraining Set Metrics (LSTM):")
    print(f"Accuracy: {lstm_train_acc:.4f}")
    print(f"Precision: {precision_score(y_train_sk, y_train_pred_lstm, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_train_sk, y_train_pred_lstm, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_train_sk, y_train_pred_lstm, average='weighted'):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_train_sk, y_train_pred_lstm)}")
    
    print("\nTest Set Metrics (LSTM):")
    print(f"Accuracy: {lstm_test_acc:.4f}")
    print(f"Precision: {precision_score(y_test_sk, y_test_pred_lstm, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test_sk, y_test_pred_lstm, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_test_sk, y_test_pred_lstm, average='weighted'):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test_sk, y_test_pred_lstm)}")
    
    # 7. GRU模型
    print("\n=== GRU Model ===")
    # 创建GRU模型
    gru_model = models.Sequential([
        layers.GRU(128, return_sequences=True, input_shape=(32, 32)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.GRU(64),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # 编译GRU模型
    gru_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 训练GRU模型
    gru_history = gru_model.fit(
        X_train_lstm,  # 使用与LSTM相同的数据格式
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 评估GRU模型
    gru_train_loss, gru_train_acc = gru_model.evaluate(X_train_lstm, y_train, verbose=0)
    gru_test_loss, gru_test_acc = gru_model.evaluate(X_test_lstm, y_test, verbose=0)
    
    # 获取预测结果
    y_train_pred_gru = np.argmax(gru_model.predict(X_train_lstm), axis=1)
    y_test_pred_gru = np.argmax(gru_model.predict(X_test_lstm), axis=1)
    
    # 打印评估指标
    print("\nTraining Set Metrics (GRU):")
    print(f"Accuracy: {gru_train_acc:.4f}")
    print(f"Precision: {precision_score(y_train_sk, y_train_pred_gru, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_train_sk, y_train_pred_gru, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_train_sk, y_train_pred_gru, average='weighted'):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_train_sk, y_train_pred_gru)}")
    
    print("\nTest Set Metrics (GRU):")
    print(f"Accuracy: {gru_test_acc:.4f}")
    print(f"Precision: {precision_score(y_test_sk, y_test_pred_gru, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test_sk, y_test_pred_gru, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_test_sk, y_test_pred_gru, average='weighted'):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test_sk, y_test_pred_gru)}")
    
    # 8. Transformer模型
    print("\n=== Transformer Model ===")
    
    # 定义Transformer编码器层
    class TransformerBlock(layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
            super(TransformerBlock, self).__init__()
            self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = tf.keras.Sequential([
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ])
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = layers.Dropout(rate)
            self.dropout2 = layers.Dropout(rate)
            
        def call(self, inputs, training=False):
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)
    
    # 定义位置编码
    class PositionalEncoding(layers.Layer):
        def __init__(self, position, d_model):
            super(PositionalEncoding, self).__init__()
            self.pos_encoding = self.positional_encoding(position, d_model)
            
        def get_angles(self, position, i, d_model):
            angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
            return position * angles
        
        def positional_encoding(self, position, d_model):
            angle_rads = self.get_angles(
                position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                d_model=d_model
            )
            # 应用正弦函数到偶数索引
            sines = tf.math.sin(angle_rads[:, 0::2])
            # 应用余弦函数到奇数索引
            cosines = tf.math.cos(angle_rads[:, 1::2])
            
            pos_encoding = tf.concat([sines, cosines], axis=-1)
            pos_encoding = pos_encoding[tf.newaxis, ...]
            return tf.cast(pos_encoding, tf.float32)
        
        def call(self, inputs):
            return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
    
    # 构建Transformer模型
    def build_transformer_model(input_shape, num_classes):
        embed_dim = 32  # 特征维度
        num_heads = 4   # 注意力头数
        ff_dim = 64     # 前馈网络维度
        
        inputs = layers.Input(shape=input_shape)
        # 添加位置编码
        positions = PositionalEncoding(input_shape[0], d_model=embed_dim)(inputs)
        # Transformer块
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(positions)
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
        # 全局平均池化
        x = layers.GlobalAveragePooling1D()(x)
        # 全连接层
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # 创建Transformer模型
    transformer_model = build_transformer_model((32, 32), num_classes)
    
    # 编译Transformer模型
    transformer_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 训练Transformer模型
    transformer_history = transformer_model.fit(
        X_train_lstm,  # 使用与LSTM相同的数据格式
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 评估Transformer模型
    transformer_train_loss, transformer_train_acc = transformer_model.evaluate(X_train_lstm, y_train, verbose=0)
    transformer_test_loss, transformer_test_acc = transformer_model.evaluate(X_test_lstm, y_test, verbose=0)
    
    # 获取预测结果
    y_train_pred_transformer = np.argmax(transformer_model.predict(X_train_lstm), axis=1)
    y_test_pred_transformer = np.argmax(transformer_model.predict(X_test_lstm), axis=1)
    
    # 打印评估指标
    print("\nTraining Set Metrics (Transformer):")
    print(f"Accuracy: {transformer_train_acc:.4f}")
    print(f"Precision: {precision_score(y_train_sk, y_train_pred_transformer, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_train_sk, y_train_pred_transformer, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_train_sk, y_train_pred_transformer, average='weighted'):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_train_sk, y_train_pred_transformer)}")
    
    print("\nTest Set Metrics (Transformer):")
    print(f"Accuracy: {transformer_test_acc:.4f}")
    print(f"Precision: {precision_score(y_test_sk, y_test_pred_transformer, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_test_sk, y_test_pred_transformer, average='weighted'):.4f}")
    print(f"F1-score: {f1_score(y_test_sk, y_test_pred_transformer, average='weighted'):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test_sk, y_test_pred_transformer)}")
    
    # 返回所有模型
    return {
        'random_forest': rf, 
        'xgboost': xgb_model, 
        'voting': voting, 
        'stacking': stacking,
        'cnn': cnn_model,
        'lstm': lstm_model,
        'gru': gru_model,
        'transformer': transformer_model
    }

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

def plot_dl_model_roc_curves(models_dict, X_test, X_test_lstm, y_test, y_test_sk):
    """
    Plot ROC curves for deep learning models
    
    Args:
        models_dict: Dictionary of trained models
        X_test: Test features for CNN
        X_test_lstm: Test features for sequence models (LSTM, GRU, Transformer)
        y_test: One-hot encoded test labels
        y_test_sk: Non one-hot encoded test labels
    """
    plt.figure(figsize=(12, 8))
    
    # Get the number of classes
    n_classes = len(np.unique(y_test_sk))
    
    # Define models to evaluate
    dl_models = {
        'CNN': (models_dict['cnn'], X_test),
        'LSTM': (models_dict['lstm'], X_test_lstm),
        'GRU': (models_dict['gru'], X_test_lstm),
        'Transformer': (models_dict['transformer'], X_test_lstm)
    }
    
    # For each model
    for model_name, (model, X) in dl_models.items():
        # Get predicted probabilities
        y_score = model.predict(X)
        
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
    plt.title('Deep Learning Models - ROC Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('dl_model_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print AUC scores for each class and each model
    print("\n=== Deep Learning Models - AUC Scores by Class ===")
    for model_name, (model, X) in dl_models.items():
        print(f"\n{model_name}:")
        
        # Get predicted probabilities
        y_score = model.predict(X)
        
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

def plot_model_comparison(models_dict, X_test, X_test_lstm, y_test_sk):
    """
    Plot accuracy comparison for all models
    
    Args:
        models_dict: Dictionary of trained models
        X_test: Test features for traditional ML and CNN
        X_test_lstm: Test features for sequence models
        y_test_sk: Non one-hot encoded test labels
    """
    # Calculate accuracy for each model
    accuracies = {}
    
    # Traditional ML models
    for name in ['random_forest', 'xgboost', 'voting', 'stacking']:
        model = models_dict[name]
        X_reshaped = X_test.reshape(X_test.shape[0], -1)
        y_pred = model.predict(X_reshaped)
        accuracies[name] = accuracy_score(y_test_sk, y_pred)
    
    # CNN model
    y_pred_cnn = np.argmax(models_dict['cnn'].predict(X_test), axis=1)
    accuracies['cnn'] = accuracy_score(y_test_sk, y_pred_cnn)
    
    # Sequence models
    for name in ['lstm', 'gru', 'transformer']:
        model = models_dict[name]
        y_pred = np.argmax(model.predict(X_test_lstm), axis=1)
        accuracies[name] = accuracy_score(y_test_sk, y_pred)
    
    # Create a bar plot
    plt.figure(figsize=(14, 8))
    
    # Define colors for different model types
    colors = {
        'random_forest': '#1f77b4',  # Traditional ML
        'xgboost': '#1f77b4',
        'voting': '#1f77b4',
        'stacking': '#1f77b4',
        'cnn': '#ff7f0e',  # CNN
        'lstm': '#2ca02c',  # Sequence models
        'gru': '#2ca02c',
        'transformer': '#2ca02c'
    }
    
    # Define model type markers
    model_types = {
        'random_forest': 'Traditional ML',
        'xgboost': 'Traditional ML',
        'voting': 'Traditional ML',
        'stacking': 'Traditional ML',
        'cnn': 'CNN',
        'lstm': 'Sequence Model',
        'gru': 'Sequence Model',
        'transformer': 'Sequence Model'
    }
    
    # Sort models by accuracy
    sorted_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    model_names = [m[0].replace('_', ' ').title() for m in sorted_models]
    accuracy_values = [m[1] for m in sorted_models]
    
    # Create bars with different colors based on model type
    bars = plt.bar(
        model_names, 
        accuracy_values,
        color=[colors[m[0]] for m in sorted_models]
    )
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f'{height:.4f}',
            ha='center', 
            va='bottom',
            fontsize=10
        )
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Traditional ML'),
        Patch(facecolor='#ff7f0e', label='CNN'),
        Patch(facecolor='#2ca02c', label='Sequence Model')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    # Set plot properties
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Accuracy Comparison', fontsize=14)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_hierarchical_ensemble_comparison(hier_ensemble, models_dict, X_test, X_test_lstm, y_test_sk):
    """
    Plot comparison of hierarchical ensemble vs. individual models
    
    Args:
        hier_ensemble: Hierarchical ensemble model dictionary
        models_dict: Dictionary of individual models
        X_test: Test features for traditional ML and CNN
        X_test_lstm: Test features for sequence models
        y_test_sk: Non one-hot encoded test labels
    """
    # Calculate accuracy for base models
    accuracies = {}
    
    # Traditional ML models
    for name in ['random_forest', 'xgboost', 'voting', 'stacking']:
        model = models_dict[name]
        X_reshaped = X_test.reshape(X_test.shape[0], -1)
        y_pred = model.predict(X_reshaped)
        accuracies[name] = accuracy_score(y_test_sk, y_pred)
    
    # CNN model
    y_pred_cnn = np.argmax(models_dict['cnn'].predict(X_test), axis=1)
    accuracies['cnn'] = accuracy_score(y_test_sk, y_pred_cnn)
    
    # Sequence models
    for name in ['lstm', 'gru', 'transformer']:
        model = models_dict[name]
        y_pred = np.argmax(model.predict(X_test_lstm), axis=1)
        accuracies[name] = accuracy_score(y_test_sk, y_pred)
    
    # Hierarchical ensemble components
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    trad_ens_pred = hier_ensemble['trad_ensemble'].predict(X_test_flat)
    accuracies['trad_ensemble'] = accuracy_score(y_test_sk, trad_ens_pred)
    
    X_test_seq = X_test.reshape(X_test.shape[0], 32, 32)
    seq_ens_pred = hier_ensemble['seq_ensemble'].predict(X_test_seq)
    accuracies['seq_ensemble'] = accuracy_score(y_test_sk, seq_ens_pred)
    
    # Final hierarchical ensemble
    final_pred = hier_ensemble['final_ensemble'].predict(hier_ensemble['x_test_meta'])
    accuracies['hierarchical_ensemble'] = accuracy_score(y_test_sk, final_pred)
    
    # Add hierarchical ensemble to the plot with special color
    plt.figure(figsize=(14, 8))
    
    # Define colors for different model types
    colors = {
        'random_forest': '#1f77b4',      # Traditional ML - Blue
        'xgboost': '#1f77b4',
        'voting': '#1f77b4',
        'stacking': '#1f77b4',
        'trad_ensemble': '#1f77b4',
        'cnn': '#ff7f0e',                # CNN - Orange
        'lstm': '#2ca02c',               # Sequence models - Green
        'gru': '#2ca02c',
        'transformer': '#2ca02c',
        'seq_ensemble': '#2ca02c',
        'hierarchical_ensemble': '#d62728'  # Hierarchical Ensemble - Red
    }
    
    # Sort models by accuracy
    sorted_models = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    model_names = [m[0].replace('_', ' ').title() for m in sorted_models]
    accuracy_values = [m[1] for m in sorted_models]
    
    # Create bars with different colors based on model type
    bars = plt.bar(
        model_names, 
        accuracy_values,
        color=[colors[m[0]] for m in sorted_models]
    )
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f'{height:.4f}',
            ha='center', 
            va='bottom',
            fontsize=10
        )
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='Traditional ML'),
        Patch(facecolor='#ff7f0e', label='CNN'),
        Patch(facecolor='#2ca02c', label='Sequence Model'),
        Patch(facecolor='#d62728', label='Hierarchical Ensemble')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    # Set plot properties
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Hierarchical Ensemble vs. Individual Models', fontsize=14)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('hierarchical_ensemble_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_hierarchical_ensemble_roc(hier_ensemble, X_test_meta, y_test, y_test_sk):
    """
    Plot ROC curve for hierarchical ensemble model
    
    Args:
        hier_ensemble: Hierarchical ensemble model dictionary
        X_test_meta: Meta features for test set
        y_test: One-hot encoded test labels
        y_test_sk: Non one-hot encoded test labels
    """
    plt.figure(figsize=(10, 8))
    
    # Get the number of classes
    n_classes = y_test.shape[1]
    
    # Get the final ensemble model
    final_model = hier_ensemble['final_ensemble']
    
    # Get predicted probabilities
    y_score = final_model.predict_proba(X_test_meta)
    
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
        
        # Plot class-specific ROC curve
        plt.plot(
            fpr[i], 
            tpr[i],
            label=f'Class {i} (AUC = {roc_auc[i]:.3f})',
            linewidth=1.5,
            alpha=0.7
        )
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot micro-average ROC curve
    plt.plot(
        fpr["micro"], 
        tpr["micro"],
        label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
        linewidth=2,
        color='deeppink',
        linestyle='-'
    )
    
    # Plot macro-average ROC curve
    plt.plot(
        fpr["macro"], 
        tpr["macro"],
        label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
        linewidth=2,
        color='navy',
        linestyle=':'
    )
    
    # Plot random guessing line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    
    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Hierarchical Ensemble - ROC Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('hierarchical_ensemble_roc.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print AUC scores for each class
    print("\n=== Hierarchical Ensemble - AUC Scores by Class ===")
    for i in range(n_classes):
        print(f"  Class {i}: {roc_auc[i]:.4f}")
    print(f"  Micro-average: {roc_auc['micro']:.4f}")
    print(f"  Macro-average: {roc_auc['macro']:.4f}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Audio classification using hierarchical ensemble')
    parser.add_argument('--data_path', type=str, default='/kaggle/input/infant-crying-dataset/infant_crying_data/train',
                        help='Path to the audio data directory')
    parser.add_argument('--force_reload', action='store_true',
                        help='Force reload and reprocess data instead of using cache')
    return parser.parse_args()

if __name__ == "__main__":
    print("Starting program...")
    # 解析命令行参数
    args = parse_args()
    print(f"Args: {args}")
    
    # 加载数据
    print("\n=== 数据加载与处理 ===")
    try:
        data_path = os.path.join(os.path.dirname(__file__), "infant_crying_data", "train")
        print(f"Data path: {data_path}")
        if not os.path.exists(data_path):
            print(f"Warning: Data path does not exist: {data_path}")
        X_train, X_test, y_train, y_test = load_and_process_data(data_path, force_reload=args.force_reload)
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 准备序列模型的数据格式
    X_train_lstm = X_train.reshape(X_train.shape[0], 32, 32)
    X_test_lstm = X_test.reshape(X_test.shape[0], 32, 32)
    y_test_sk = np.argmax(y_test, axis=1)
    
    # 训练和评估基础模型
    print("\n=== 训练和评估基础模型 ===")
    models_dict = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # 构建层次化集成模型
    print("\n=== 构建层次化集成模型 ===")
    hier_ensemble = build_hierarchical_ensemble(
        models_dict, X_train, X_test, y_train, y_test, X_train_lstm, X_test_lstm
    )
    
    # 绘制传统机器学习模型的ROC曲线
    print("\n=== 绘制传统机器学习模型ROC曲线 ===")
    ml_models = [models_dict['random_forest'], models_dict['xgboost'], 
                models_dict['voting'], models_dict['stacking']]
    ml_model_names = ['Random Forest', 'XGBoost', 'Voting Ensemble', 'Stacking Ensemble']
    plot_roc_curves(ml_models, X_test, y_test_sk, ml_model_names)
    
    # 绘制深度学习模型的ROC曲线
    print("\n=== 绘制深度学习模型ROC曲线 ===")
    plot_dl_model_roc_curves(models_dict, X_test, X_test_lstm, y_test, y_test_sk)
    
    # 绘制层次化集成模型的ROC曲线
    print("\n=== 绘制层次化集成模型ROC曲线 ===")
    plot_hierarchical_ensemble_roc(hier_ensemble, hier_ensemble['x_test_meta'], y_test, y_test_sk)
    
    # 绘制所有模型的准确率比较
    print("\n=== 绘制所有模型准确率比较 ===")
    plot_model_comparison(models_dict, X_test, X_test_lstm, y_test_sk)
    
    # 绘制层次化集成与其他模型的比较
    print("\n=== 绘制层次化集成与其他模型的比较 ===")
    plot_hierarchical_ensemble_comparison(hier_ensemble, models_dict, X_test, X_test_lstm, y_test_sk)
    
    print("\n=== 模型训练和评估完成 ===")