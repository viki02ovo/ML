import bs4
import requests
import pandas as pd
import re
import numpy as np
import os
import lxml


def get_sp500_sector_data():
    """获取S&P 500公司的行业数据"""
    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    resp = requests.get(wiki_url, headers=headers)
    soup = bs4.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'id': 'constituents'})

    # 提取数据
    data = []
    for row in table.findAll('tr')[1:]:
        cells = row.find_all('td')
        if len(cells) >= 4:
            ticker = cells[0].text.strip()
            company = cells[1].text.strip()
            sector = cells[2].text.strip()
            sub_industry = cells[3].text.strip()
            data.append([ticker, company, sector, sub_industry])
            print(data)

    # 创建DataFrame
    df_sp500 = pd.DataFrame(data, columns=['Ticker', 'CompanyName', 'Sector', 'sub_industry'])
    return df_sp500


def batch_add_sector(output_suffix='_with_sector', sp500=None):
    """
    批量为多个CSV文件添加Sector列

    参数:
    csv_files_list: CSV文件路径列表
    output_suffix: 输出文件后缀
    """
    # 获取一次S&P 500数据，供所有文件使用
    sector_dict = dict(zip(sp500['Ticker'], sp500['Sector']))
    results = []
    output_file_path = ''
    for dirname, _, filenames in os.walk('/data/S&P500/individual_stocks_5yr'):
        for filename in filenames:
            csv_file = os.path.join(dirname, filename)
            print(csv_file)
            # 读取CSV
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()

            # 检查Name列是否存在
            if 'Name' not in df.columns:
                print(f"  警告: {csv_file} 中没有Name列，跳过")
                continue

            # 添加Sector列
            stock_name = df['Name'].iloc[0]
            df['Sector'] = df['Name'].map(sector_dict)

            if df['Sector'].iloc[0] is not None:
                print(f"成功添加Sector: {df['Sector'].iloc[0]}")
            else:
                print(f"未找到股票: {stock_name}")

            # 保存文件
            output_file = csv_file.replace('.csv', f'{output_suffix}.csv')
            df.to_csv(output_file, index=False)
            print(f"保存到: {output_file}")

            results.append(df)
    return results


def main():
    print("=" * 60)
    print("S&P 500股票行业信息批量添加工具")
    print("=" * 60)
    try:
        df_sp = get_sp500_sector_data()
        print(f"✓ 成功获取 {len(df_sp)} 家公司的行业数据")
        print(f"  示例数据:\n{df_sp.head()}")
        # # 可选：保存S&P 500数据到本地
        # sp500_output = 'sp500_sectors.csv'
        # df_sp.to_csv(sp500_output, index=False)
        # print(f"  S&P 500数据已保存至: {sp500_output}")
    except Exception as e:
        print(f"✗ 获取S&P 500数据失败: {e}")
        return

    # 步骤2: 批量为股票CSV文件添加行业信息
    # 检查数据目录是否存在
    data_dir = '/data/S&P500/individual_stocks_5yr'
    if not os.path.exists(data_dir):
        print(f"✗ 数据目录不存在: {data_dir}")
        print("  请确保数据文件存放在正确的路径下")
        return

    try:
        processed_files = batch_add_sector(output_suffix='_with_sector', sp500=df_sp)
        print(f"\n✓ 成功处理 {len(processed_files)} 个文件")
    except Exception as e:
        print(f"✗ 批量处理过程中出错: {e}")
        return

    # 步骤3: 输出处理结果统计
    print("\n[步骤3] 处理结果统计")
    print("-" * 60)
    print(f"处理文件总数: {len(processed_files)}")
    if processed_files:
        print(f"输出文件后缀: _with_sector.csv")
        print(f"处理完成！所有文件已保存到原目录")

    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()



