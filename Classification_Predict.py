import os

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


# ------------------------------
# 1. 选择代表性股票（从聚类结果中选择）
# ------------------------------
def select_representative_stocks():
    """从聚类结果中每个cluster选择1-2只股票"""
    cluster_results = pd.read_csv("result/cluster_results.csv")

    # 从每个cluster选择股票
    selected_stocks = []
    for cluster in cluster_results['cluster'].unique():
        cluster_stocks = cluster_results[cluster_results['cluster'] == cluster]
        # 选择夏普比率最高的1-2只股票
        top_stocks = cluster_stocks.nlargest(2, 'sharpe_ratio')['ticker'].tolist()
        selected_stocks.extend(top_stocks)

    # 确保至少有5只股票，如果不够则添加知名股票
    if len(selected_stocks) < 5:
        famous_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        for stock in famous_stocks:
            if stock not in selected_stocks:
                selected_stocks.append(stock)
            if len(selected_stocks) >= 5:
                break

    return selected_stocks[:8]  # 最多选择8只股票


# ------------------------------
# 2. 特征工程：技术指标
# ------------------------------
def calculate_technical_indicators(df):
    """计算技术指标作为特征"""
    # 移动平均线
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # 相对强弱指标(RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()

    # 布林带
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # 价格变化率
    df['Price_change_1d'] = df['Close'].pct_change(1)
    df['Price_change_5d'] = df['Close'].pct_change(5)
    df['Price_change_10d'] = df['Close'].pct_change(10)

    # 成交量相关
    df['Volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

    # 波动率
    df['Volatility_5d'] = df['Close'].rolling(window=5).std()
    df['Volatility_20d'] = df['Close'].rolling(window=20).std()

    return df


# ------------------------------
# 3. 创建标签（趋势预测目标）
# ------------------------------
def create_trend_labels(df, prediction_days=5):
    """
    创建趋势标签：
    0: 下跌 (未来N天收益率 < -2%)
    1: 持平 (-2% <= 未来N天收益率 <= 2%)
    2: 上涨 (未来N天收益率 > 2%)
    """
    future_return = df['Close'].pct_change(prediction_days).shift(-prediction_days)

    conditions = [
        future_return < -0.02,  # 下跌超过2%
        (future_return >= -0.02) & (future_return <= 0.02),  # 持平
        future_return > 0.02  # 上涨超过2%
    ]

    df['trend_label'] = np.select(conditions, [0, 1, 2], default=np.nan)
    return df


# ------------------------------
# 4. 数据准备函数
# ------------------------------
def prepare_data_for_prediction(ticker, start_date='2020-01-01', end_date='2024-12-31'):
    """为单只股票准备预测数据"""
    try:
        # 下载数据
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty or len(df) < 100:
            return None, None

        # 计算技术指标
        df = calculate_technical_indicators(df)

        # 创建标签
        df = create_trend_labels(df)

        # 删除NaN值
        df = df.dropna()

        if len(df) < 50:
            return None, None

        # 选择特征
        feature_columns = [
            'SMA_5', 'SMA_10', 'SMA_20', 'RSI', 'MACD', 'MACD_signal',
            'BB_position', 'Price_change_1d', 'Price_change_5d', 'Price_change_10d',
            'Volume_ratio', 'Volatility_5d', 'Volatility_20d'
        ]

        X = df[feature_columns]
        y = df['trend_label']

        return X, y

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None, None


# ------------------------------
# 5. 模型训练和评估
# ------------------------------
def train_and_evaluate_models(X, y, ticker, save_plots=True):
    """训练多个模型并评估"""
    # # 分割数据
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42, stratify=y
    # )
    # 计算分割点
    split_index = int(len(X) * 0.8)

    # 按时间顺序切分：前80%训练，后20%测试
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    }

    results = {}
    class_names = ['Down', 'Flat', 'Up']

    for name, model in models.items():
        # 训练模型
        model.fit(X_train_scaled, y_train)
        # 预测
        y_pred = model.predict(X_test_scaled)
        # 评估
        accuracy = accuracy_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred,
            'y_test': y_test,
            'scaler': scaler
        }

        print(f"\n{ticker} - {name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))

        # ---------- 可视化部分 ----------
        if save_plots:
            # 创建保存目录
            plot_dir = f"result/{ticker}/{name.replace(' ', '_')}"
            os.makedirs(plot_dir, exist_ok=True)

            # 1. 混淆矩阵热图
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'{ticker} - {name} Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/confusion_matrix.png", dpi=150)
            plt.close()

            # 2. 特征重要性（仅树模型，XGBoost 也有 feature_importances_）
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                plt.figure(figsize=(10, 6))
                sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
                plt.title(f'{ticker} - {name} Feature Importance')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.tight_layout()
                plt.savefig(f"{plot_dir}/feature_importance.png", dpi=150)
                plt.close()

            # 3. 分类报告柱状图
            report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
            metrics_df = pd.DataFrame(report).transpose().drop(['accuracy', 'macro avg', 'weighted avg'],
                                                               errors='ignore')
            metrics_df = metrics_df[['precision', 'recall', 'f1-score']]

            metrics_df.plot(kind='bar', figsize=(10, 6))
            plt.title(f'{ticker} - {name} Classification Metrics by Class')
            plt.xlabel('Class')
            plt.ylabel('Score')
            plt.xticks(rotation=0)
            plt.legend(loc='lower right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/classification_metrics.png", dpi=150)
            plt.close()

    return results


# ------------------------------
# 6. 主程序：趋势预测
# ------------------------------
def main_trend_prediction():
    print("=== 股价趋势预测任务 ===")

    # 选择代表性股票
    selected_stocks = select_representative_stocks()
    print(f"选择的股票: {selected_stocks}")

    all_results = {}

    # 为每只股票训练模型
    for ticker in tqdm(selected_stocks, desc="Training models"):
        print(f"\n处理股票: {ticker}")

        X, y = prepare_data_for_prediction(ticker)

        if X is None or len(X) < 50:
            print(f"跳过 {ticker}: 数据不足")
            continue

        # 检查标签分布
        print(f"标签分布: {y.value_counts().to_dict()}")

        if len(y.unique()) < 2:
            print(f"跳过 {ticker}: 标签类别不足")
            continue

        # 训练和评估模型
        results = train_and_evaluate_models(X, y, ticker)
        all_results[ticker] = results

    return all_results


# ------------------------------
# 7. 改进1：集成学习
# ------------------------------
def improvement_1_ensemble():
    """改进1: 使用更复杂的集成学习方法"""
    print("\n=== 改进1: 集成学习 ===")

    selected_stocks = select_representative_stocks()

    for ticker in selected_stocks:
        print(f"\n处理股票: {ticker}")
        X, y = prepare_data_for_prediction(ticker)

        if X is None:
            continue

        # # 分割数据
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.2, random_state=42, stratify=y
        # )
        # 计算分割点
        split_index = int(len(X) * 0.8)

        # 按时间顺序切分：前80%训练，后20%测试
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 创建集成模型
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        svm = SVC(probability=True, random_state=42)

        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
            voting='soft'
        )

        # 训练集成模型
        ensemble.fit(X_train_scaled, y_train)
        y_pred_ensemble = ensemble.predict(X_test_scaled)

        accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
        print(f"集成模型准确率: {accuracy_ensemble:.4f}")


# ------------------------------
# 8. 改进2：特征选择优化
# ------------------------------
def improvement_2_feature_selection():
    """改进2: 特征选择和超参数优化"""
    print("\n=== 改进2: 特征选择优化 ===")

    selected_stocks = select_representative_stocks()

    for ticker in selected_stocks:
        print(f"\n处理股票: {ticker}")
        X, y = prepare_data_for_prediction(ticker)

        if X is None:
            continue

        # # 分割数据
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.2, random_state=42, stratify=y
        # )
        # 计算分割点
        split_index = int(len(X) * 0.8)

        # 按时间顺序切分：前80%训练，后20%测试
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        # 创建pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=8)),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # 参数网格
        param_grid = {
            'feature_selection__k': [6, 8, 10],
            'classifier__n_estimators': [50, 100, 150],
            'classifier__max_depth': [5, 10, None]
        }

        # 网格搜索
        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        y_pred_optimized = grid_search.predict(X_test)

        accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
        print(f"优化后准确率: {accuracy_optimized:.4f}")
        print(f"最佳参数: {grid_search.best_params_}")


# ------------------------------
# 主执行
# ------------------------------
if __name__ == "__main__":
    # 执行趋势预测主任务
    trend_results = main_trend_prediction()

    # 执行改进1
    improvement_1_ensemble()

    # 执行改进2
    improvement_2_feature_selection()

    print("\n=== 项目完成总结 ===")
    print("✓ 任务1: 股票聚类 (已完成)")
    print("✓ 任务2: 价格趋势预测 (已完成)")
    print("✓ 改进1: 集成学习方法")
    print("✓ 改进2: 特征选择和超参数优化")
