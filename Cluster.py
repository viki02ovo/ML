import os
import bs4
import pandas as pd
import numpy as np
import requests
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import requests
from bs4 import BeautifulSoup
from sklearn.decomposition import PCA
import seaborn as sns


# ------------------------------
# 1. 获取S&P 500股票列表
# ------------------------------
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, 'lxml')

    # 尝试多种选择器
    table = soup.find('table', {'id': 'constituents'})
    if table is None:
        # 备用：查找第一个wikitable类名的表格
        table = soup.find('table', {'class': 'wikitable'})

    if table is None:
        raise ValueError("Could not find S&P 500 table on Wikipedia")

    original_tickers = []
    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        if cells:
            ticker = cells[0].text.strip()
            if ticker:
                original_tickers.append(ticker)

    cleaned_tickers = [ticker.replace('.', '-') for ticker in original_tickers]
    return cleaned_tickers


# ------------------------------
# 2. 下载数据并计算单只股票的特征
# ------------------------------
def compute_stock_features(ticker, start_date='2015-01-01', end_date='2024-12-31'):
    """
    下载单只股票的历史日线数据，并计算用于聚类的特征
    返回一个字典（特征）
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df.empty or len(df) < 50:  # 需要足够数据
            return None

        data_dir = "stock_data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        csv_path = os.path.join(data_dir, f"{ticker}.csv")
        df.to_csv(csv_path)

        # 计算每日收益率
        df['Return'] = df['Close'].pct_change()
        df = df.dropna()

        if len(df) == 0:
            return None

        # 年化因子（252个交易日）
        annual_factor = 252

        # 特征1：年化平均收益率
        mean_daily_return = df['Return'].mean()
        annual_return = (1 + mean_daily_return) ** annual_factor - 1

        # 特征2：年化波动率
        daily_vol = df['Return'].std()
        annual_vol = daily_vol * np.sqrt(annual_factor)

        # 特征3：夏普比率（假设无风险利率为0）
        sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0

        # 特征4：最大回撤
        cumulative = (1 + df['Return']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # 特征5：平均成交量（标准化用对数）
        avg_volume = df['Volume'].mean()
        log_avg_volume = np.log(avg_volume + 1)

        # 特征6：偏度（收益率分布的对称性）
        skewness = df['Return'].skew()

        # 特征7：峰度（尾部风险）
        kurtosis = df['Return'].kurtosis()

        # 可选：最近一年的表现（最近252天收益率）
        recent_return = (df['Close'].iloc[-1] / df['Close'].iloc[-252] - 1) if len(df) >= 252 else np.nan

        features = {
            'ticker': ticker,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'log_avg_volume': log_avg_volume,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'recent_1y_return': recent_return,
            'data_points': len(df)  # 用于质量控制
        }
        return features
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None


# ------------------------------
# 3. 批量计算所有股票的特征
# ------------------------------
def build_feature_matrix(tickers, start_date, end_date):
    """循环所有股票，计算特征，返回DataFrame"""
    features_list = []
    for ticker in tqdm(tickers, desc="Downloading and computing features"):
        feat = compute_stock_features(ticker, start_date, end_date)
        if feat is not None:
            features_list.append(feat)
        time.sleep(0.1)  # 避免请求过快被限制
    features_df = pd.DataFrame(features_list)
    return features_df


# ------------------------------
# 4. 主程序：执行聚类
# ------------------------------
if __name__ == "__main__":
    # print("Step 1: Getting S&P 500 tickers...")
    # tickers = get_sp500_tickers()
    # print(f"Total tickers: {len(tickers)}")
    #
    # print("Step 2: Downloading data and computing features...")
    # feature_df = build_feature_matrix(tickers, start_date='2020-01-01', end_date='2024-12-31')

    # # 去除含有NaN的行（某些特征可能计算失败）
    # feature_df = feature_df.dropna()
    # print(f"Successfully computed features for {len(feature_df)} stocks")
    #
    result_dir = "result"
    # if not os.path.exists(result_dir):
    #     os.makedirs(result_dir)


    # # 保存特征矩阵到CSV
    # feature_df.to_csv(os.path.join(result_dir, "stock_features.csv"), index=False)
    # print("Saved features to path: result/stock_features.csv")

    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_dir, 'result/stock_features.csv')

    print("\n Loading data...")
    feature_df = pd.read_csv(data_path)


    # 选择用于聚类的数值列進行標準化（排除 ticker 和 data_points）
    cluster_cols = ['annual_return', 'annual_volatility', 'sharpe_ratio',
                    'max_drawdown', 'log_avg_volume', 'skewness', 'kurtosis', 'recent_1y_return']
    X = feature_df[cluster_cols].copy()

    # 标准化（重要，因为不同特征量纲差异大）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 确定最佳聚类数（使用肘部法则，可选）
    # inertias = []
    # K_range = range(2, 11)
    # for k in K_range:
    #     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    #     kmeans.fit(X_scaled)
    #     inertias.append(kmeans.inertia_)
    #
    # # 绘制肘部图
    # plt.figure(figsize=(8, 5))
    # plt.plot(K_range, inertias, 'bo-')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Inertia')
    # plt.title('Elbow Method for Optimal K')
    # plt.grid(True)
    # plt.savefig('elbow_plot.png')
    # plt.show()

    # 选择聚類大小 K
    n_clusters = 4
    print(f"Using n_clusters = {n_clusters}")

    # 最终聚类
    kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    feature_df['cluster'] = kmeans_final.fit_predict(X_scaled)

    # 保存带有聚类标签的结果
    result_df = feature_df[['ticker', 'cluster'] + cluster_cols]
    result_df.to_csv(os.path.join(result_dir, "cluster_results.csv"), index=False)
    print("Saved clustering results to path: result/cluster_results.csv")

    # 显示每个聚类的股票数量
    print("\nCluster distribution:")
    print(result_df['cluster'].value_counts().sort_index())

    # 输出每个聚类的特征均值（便于解释）
    cluster_summary = result_df.groupby('cluster')[cluster_cols].mean()
    print("\nCluster characteristics (mean values):")
    print(cluster_summary)

    # ==============================
    # 可视化聚类结果
    # ==============================
    print("\nGenerating visualization plots...")

    # 设置 seaborn 风格
    sns.set_style("whitegrid")
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签（如有需要）
    plt.rcParams['axes.unicode_minus'] = False

    # --- 图1：PCA降维后的二维散点图 ---
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['cluster'] = feature_df['cluster'].values
    pca_df['ticker'] = feature_df['ticker'].values

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'],
                          c=pca_df['cluster'], cmap='viridis', alpha=0.7, edgecolors='w', s=60)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('PCA Visualization of Stock Clusters')
    plt.colorbar(scatter, label='Cluster')
    # 可选：标注部分极端点（如远离中心的股票）
    for i, row in pca_df.iterrows():
        if abs(row['PC1']) > 5 or abs(row['PC2']) > 5:
            plt.annotate(row['ticker'], (row['PC1'], row['PC2']), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "cluster_pca_scatter.png"), dpi=300)
    # plt.show()

    # --- 图2：聚类特征均值雷达图 ---
    # 准备雷达图数据：将各聚类均值归一化到0-1之间，便于比较
    from sklearn.preprocessing import MinMaxScaler
    radar_scaler = MinMaxScaler()
    cluster_scaled = pd.DataFrame(
        radar_scaler.fit_transform(cluster_summary),
        columns=cluster_summary.columns,
        index=cluster_summary.index
    )

    # 雷达图需要闭合数据，添加第一个特征到末尾
    categories = cluster_scaled.columns.tolist()
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for i, cluster_id in enumerate(cluster_scaled.index):
        values = cluster_scaled.loc[cluster_id].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {cluster_id}')
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title('Cluster Characteristics Radar Chart (Normalized Mean Values)', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "cluster_radar.png"), dpi=300)
    # plt.show()

    # --- 圖3：箱线图 ---
    # 选择要展示的特征（可根据需要调整）
    box_features = ['annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, feat in enumerate(box_features):
        ax = axes[idx]
        # 使用 seaborn 绘制箱线图，按聚类分组
        sns.boxplot(x='cluster', y=feat, data=result_df, ax=ax, palette='Set2')
        ax.set_title(f'Distribution of {feat} by Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel(feat)
        # 可选：添加散点图抖动显示数据点分布（避免重叠严重时可注释掉）
        sns.stripplot(x='cluster', y=feat, data=result_df, ax=ax, color='black', alpha=0.3, size=2, jitter=0.2)

    plt.suptitle('Boxplots of Key Features Across Clusters', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "cluster_boxplots.png"), dpi=300, bbox_inches='tight')
    # plt.show()

    # --- 图4：聚类股票数量条形图 ---
    cluster_counts = result_df['cluster'].value_counts().sort_index()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Stocks')
    plt.title('Stock Count per Cluster')
    for i, v in enumerate(cluster_counts.values):
        plt.text(i, v + 1, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "cluster_counts.png"), dpi=300)
    # plt.show()

    print(f"Visualization plots saved to '{result_dir}' folder.")