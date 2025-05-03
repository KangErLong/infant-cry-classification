import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import tensorflow as tf
import gc
from tensorflow.keras import layers, models, Model, callbacks
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子保证可复现性
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==================== 增强型数据处理模块 ====================
def extract_audio_features(y, sr):
    """提取增强的音频特征集"""
    features = []
    
    # 基础时域特征
    features.append(np.mean(y))
    features.append(np.std(y))
    features.append(np.max(y) - np.min(y))
    features.append(np.mean(np.abs(np.diff(y))))
    # 频域特征 - MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_processed = np.mean(mfcc.T, axis=0)
    features.extend(mfcc_processed)
    
    # 频域特征 - 频谱质心
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(cent))
    
    # 频域特征 - 频谱带宽
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(bandwidth))# 频域特征 - 频谱滚降
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))
    
    # 谐波与噪声特征
    try:
        hnr = librosa.effects.harmonic(y)
        features.append(np.mean(hnr[:min(len(hnr), len(y))]))
    except:
        features.append(0)
    
    # 小波变换特征
    try:
        cqt = np.abs(librosa.cqt(y, sr=sr))
        cqt_feature = np.mean(cqt, axis=1)
        features.extend(cqt_feature[:20])  # 取前20个特征
    except:
        features.extend([0] * 20)
    
    # 色度特征
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    
    # 梅尔频谱
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel, axis=1)
    features.extend(mel_mean[:20])  # 取前20个特征return np.array(features)

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
        pass# 添加不同级别的高斯噪声
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
        return (data['X_train'], data['X_test'], 
                data['y_train'], data['y_test'])
    
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
                if len(y) < sr *0.5:  # 跳过过短的音频
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
    
    # 转换为numpy数组
    X = np.array(features)
    y = np.array(labels)# 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 转换为one-hot编码并确保形状正确
    num_classes = len(np.unique(y_encoded))
    y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)
    # 确保y_onehot是2D数组用于训练
    if len(y_onehot.shape) == 1:
        y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)
    # 确保y_onehot形状正确
    if len(y_onehot.shape) != 2:
        y_onehot = np.reshape(y_onehot, (-1, num_classes))
    # 确保y_encoded是1D数组用于stratify
    if len(y_encoded.shape) > 1:
        y_encoded = np.argmax(y_encoded, axis=1)
    
    # 分割数据并保持one-hot编码
    # 正确使用原始标签进行分层抽样
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=test_size, 
        stratify=y_encoded, random_state=SEED
    )
    
    # 调试输出标签形状
    print(f'\n=== 标签形状验证 ===')
    print(f'原始标签形状: {y_onehot.shape}')
    print(f'训练集标签形状: {y_train.shape}')
    print(f'测试集标签形状: {y_test.shape}')
    print(f'类别数量验证: {num_classes}')
    # 确保训练和测试标签为one-hot编码的(样本数, 6)形状
    if y_train.shape[1] != 6:
        raise ValueError(f"训练标签形状应为(样本数, 6)，但得到 {y_train.shape}")
    if y_test.shape[1] != 6:
        raise ValueError(f"测试标签形状应为(样本数, 6)，但得到 {y_test.shape}")
    
    # 验证形状
    print(f"One-hot encoded labels shape: {y_onehot.shape}")
    print(f"Number of classes: {num_classes}")
    
    # 验证形状
    print(f"One-hot encoded labels shape: {y_onehot.shape}")
    print(f"Number of classes: {y_onehot.shape[1]}")
    
    # 保存处理后的数据到缓存
    np.savez_compressed(
        cache_file, 
        X_train=X_train, X_test=X_test, 
        y_train=y_train, y_test=y_test
    )
    return X_train, X_test, y_train, y_test

# ==================== 改进模型构建模块 ====================
def build_enhanced_hybrid_transformer(input_shape, num_classes):
    """增强型混合Transformer模型 - 包含残差连接和自注意力机制"""
    # 输入层
    inputs = layers.Input(shape=input_shape)
    
    # 特征归一化
    normalized = layers.LayerNormalization()(inputs)
    
    # ===== CNN特征提取路径 =====
    # 重塑以用于卷积
    x_cnn = layers.Reshape((input_shape[0], 1))(normalized)
    # 多尺度卷积块1
    conv1_1 = layers.Conv1D(128, 3, padding='same')(x_cnn)
    conv1_2 = layers.Conv1D(128, 5, padding='same')(x_cnn)
    conv1_3 = layers.Conv1D(128, 7, padding='same')(x_cnn)
    
    # 连接不同内核大小的卷积结果
    conv1_concat = layers.Concatenate()([conv1_1, conv1_2, conv1_3])
    conv1_bn = layers.BatchNormalization()(conv1_concat)
    conv1_act = layers.LeakyReLU(alpha=0.1)(conv1_bn)
    conv1_drop = layers.SpatialDropout1D(0.2)(conv1_act)
    
    # 添加注意力机制
    attention_layer1 = layers.MultiHeadAttention(
        num_heads=8, key_dim=48, dropout=0.1
    )(conv1_drop, conv1_drop)
    
    # 残差连接 + 层归一化
    res1 = layers.Add()([conv1_drop, attention_layer1])
    res1_norm = layers.LayerNormalization()(res1)
    
    # 多尺度卷积块 2
    conv2_1 = layers.SeparableConv1D(256, 3, padding='same')(res1_norm)
    conv2_2 = layers.SeparableConv1D(256, 5, padding='same')(res1_norm)
    
    # 连接
    conv2_concat = layers.Concatenate()([conv2_1, conv2_2])
    conv2_bn = layers.BatchNormalization()(conv2_concat)
    conv2_act = layers.LeakyReLU(alpha=0.1)(conv2_bn)
    
    # 全局池化
    global_max = layers.GlobalMaxPooling1D()(conv2_act)
    global_avg = layers.GlobalAveragePooling1D()(conv2_act)
    cnn_features = layers.Concatenate()([global_max, global_avg])# ===== RNN特征提取路径 =====
    x_rnn = layers.Reshape((input_shape[0], 1))(normalized)
    
    # 双向LSTM层
    lstm1 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x_rnn)
    lstm1_drop = layers.SpatialDropout1D(0.2)(lstm1)
    
    # 残差连接
    lstm2 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(lstm1_drop)
    lstm_res = layers.Add()([lstm1, lstm2])
    # 注意力层
    lstm_attn = layers.MultiHeadAttention(
        num_heads=8, key_dim=32, dropout=0.1
    )(lstm_res, lstm_res)
    
    # 残差连接 + 层归一化
    lstm_res2 = layers.Add()([lstm_res, lstm_attn])
    lstm_norm = layers.LayerNormalization()(lstm_res2)
    
    # 全局池化
    lstm_pool_max = layers.GlobalMaxPooling1D()(lstm_norm)
    lstm_pool_avg = layers.GlobalAveragePooling1D()(lstm_norm)
    lstm_features = layers.Concatenate()([lstm_pool_max, lstm_pool_avg])
    
    # ===== Transformer特征提取路径 =====
    # 位置编码
    x_trans = layers.Reshape((input_shape[0], 1))(normalized)
    
    # 位置编码层
    pos_encoding = tf.range(start=0, limit=float(input_shape[0]), delta=1.0)
    pos_encoding = tf.expand_dims(tf.expand_dims(pos_encoding, axis=0), axis=-1)
    x_trans_pos = layers.Add()([x_trans, pos_encoding])
    
    # 线性投影层
    x_trans = layers.Conv1D(256, 1)(x_trans_pos)
    
    # 多层Transformer编码块
    for i in range(4):  # 4个Transformer层
        # 多头注意力
        attn_output = layers.MultiHeadAttention(
            num_heads=8, key_dim=32, dropout=0.1
        )(x_trans, x_trans)# 残差连接 + 层归一化
        x_trans = layers.Add()([x_trans, attn_output])
        x_trans = layers.LayerNormalization()(x_trans)
        
        # 前馈神经网络
        ffn = layers.Dense(512, activation='relu')(x_trans)
        ffn = layers.Dropout(0.1)(ffn)
        ffn = layers.Dense(256)(ffn)
        # 残差连接 + 层归一化
        x_trans = layers.Add()([x_trans, ffn])
        x_trans = layers.LayerNormalization()(x_trans)
    # 全局池化
    trans_pool = layers.GlobalAveragePooling1D()(x_trans)
    # ===== 特征融合 =====
    # 连接所有特征
    all_features = layers.Concatenate()([cnn_features, lstm_features, trans_pool])# 特征交互层
    fusion = layers.Dense(512, activation='relu')(all_features)
    fusion = layers.BatchNormalization()(fusion)
    fusion = layers.Dropout(0.3)(fusion)
    # 特征压缩层
    fusion = layers.Dense(256, activation='relu')(fusion)
    fusion = layers.BatchNormalization()(fusion)
    fusion = layers.Dropout(0.2)(fusion)
    
    # 输出层
    outputs = layers.Dense(num_classes, activation='softmax')(fusion)# 创建模型
    model = Model(inputs=inputs, outputs=outputs)
    
    # 优化器设置 - 使用Lookahead包装的AdamW
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=3000,
        alpha=1e-5# 最小学习率比例
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    # 编译模型 - 修复：使用与one-hot编码标签兼容的指标
    model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
    )
    return model

# ==================== 模型训练与评估模块 ====================
def main():
    # 参数设置
    DATA_PATH = "infant_crying_data/train"  # 修改为实际数据集路径
    MAX_LEN = 3                          # 音频最大长度(秒)
    SAMPLE_RATE = 16000                   # 采样率
    
    # 加载并预处理数据
    X_train, X_test, y_train, y_test = load_and_process_data(
        DATA_PATH, 
        max_len=MAX_LEN, 
        sr=SAMPLE_RATE
    )
    
    # 确保标签形状正确
    if len(y_train.shape) == 1:
        y_train = tf.keras.utils.to_categorical(y_train)
    if len(y_test.shape) == 1:
        y_test = tf.keras.utils.to_categorical(y_test)
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 获取输入形状和类别数
    input_shape = (X_train_scaled.shape[1],)
    num_classes = y_train.shape[1] if len(y_train.shape) > 1 else len(np.unique(y_train))
    print(f"\n输入特征维度: {input_shape}, 类别数: {num_classes}")

    # 构建模型
    model = build_enhanced_hybrid_transformer(input_shape, num_classes)
    
    # 定义回调函数
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    model_checkpoint = callbacks.ModelCheckpoint(
        "best_model.h5",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        save_format='h5'
    )
    
    lr_monitor = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # 训练模型
    print("\n开始训练模型...")
    history = model.fit(
        X_train_scaled,
        y_train,
        batch_size=64,
        epochs=100,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint, lr_monitor],
        verbose=1
    )

    # 加载最佳模型
    model.load_weights("best_model.h5")

    # 模型评估
    print("\n评估测试集性能:")
    y_pred = model.predict(X_test_scaled)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # 计算评估指标
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    precision = precision_score(y_test_labels, y_pred_labels, average='macro')
    recall = recall_score(y_test_labels, y_pred_labels, average='macro')
    f1 = f1_score(y_test_labels, y_pred_labels, average='macro')

    print(f"准确率: {accuracy:.4f}")
    print(f"宏平均精确率: {precision:.4f}")
    print(f"宏平均召回率: {recall:.4f}")
    print(f"宏平均F1值: {f1:.4f}")

    # 绘制混淆矩阵
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # 绘制模型结构图
    from tensorflow.keras.utils import plot_model
    try:
        plot_model(
            model, to_file='model_architecture.png',
            show_shapes=True, show_layer_names=True,
            rankdir='TB', expand_nested=False, dpi=96
        )
    except Exception as e:
        print('\n无法生成模型架构图，因为Graphviz未安装或配置不正确。')

    # 绘制学习曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率曲线')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失曲线')
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.show()

if __name__ == "__main__":
    main()
    