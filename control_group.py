import pandas as pd
import glob
import os

# 1. 設定資料夾路徑
data_dir = r'd:\apps\python-fm901\data'

# 2. 使用 glob 取得資料夾底下所有的 .csv 檔案路徑
all_files = glob.glob(os.path.join(data_dir, "*.csv"))

# 用來存放每個讀取進來的 DataFrame 的串列
df_list = []

for file in all_files:
    try:
        # 3. 讀取 CSV 檔案
        # 提示：如果遇到「UnicodeDecodeError」編碼問題，可以將 encoding 改為 'cp950' (Big5) 或 'utf-8-sig'
        df = pd.read_csv(file, sep='\t', encoding='utf-8')

        # 將「年月日」欄位轉換為日期格式 (yyyyMMdd)
        if '年月日' in df.columns:
            df['年月日'] = pd.to_datetime(
                df['年月日'].astype(str), format='%Y%m%d').dt.date

        df_list.append(df)
        print(f"成功讀取: {os.path.basename(file)}")

    except Exception as e:
        print(f"讀取檔案 {os.path.basename(file)} 時發生錯誤: {e}")

# 4. 將所有的 DataFrame 合併成一個單一的 DataFrame
if df_list:
    # ignore_index=True 確保合併後的 index 會重新從 0 開始排序
    combined_df = pd.concat(df_list, ignore_index=True)

    print("\n================ 合併完成 ================")
    print(f"總共讀取了 {len(all_files)} 個檔案")
    print(f"合併後的總資料筆數: {len(combined_df)} 筆")
    
    if '年月日' in combined_df.columns:
        min_date = combined_df['年月日'].min()
        max_date = combined_df['年月日'].max()
        total_days = (max_date - min_date).days + 1
        print(f"資料起始日期: {min_date}")
        print(f"資料結束日期: {max_date}")
        print(f"資料跨越總天數: {total_days} 天")

    # 顯示基本敘述統計表
    print("\n================ 敘述統計表 ================")
    # 解決 pandas 中文對齊問題並設定小數點位數與不使用科學記號
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    pd.set_option('display.width', 1000)

    desc = combined_df.describe()
    # 轉為字串呈現，設定小數點固定兩位且數值靠右
    print(desc.to_string(float_format=lambda x: f'{x:>.2f}', justify='right'))

    # 取得數值欄位並刪除空值資料列
    num_cols = combined_df.select_dtypes(include=['number']).columns
    original_len = len(combined_df)
    cleaned_df = combined_df.dropna(subset=num_cols)
    dropped_count = original_len - len(cleaned_df)

    print(
        f"\n================ 刪除空值後的敘述統計表 (刪除了 {dropped_count} 筆資料列) ================")
    desc_cleaned = cleaned_df.describe()
    print(desc_cleaned.to_string(
        float_format=lambda x: f'{x:>.2f}', justify='right'))

    # 進行 Winsorization (1% 及 99%)
    print("\n================ Winsorization 後的敘述統計表 (1%, 99%) ================")
    winsorized_df = cleaned_df.copy()
    for col in num_cols:
        lower_bound = winsorized_df[col].quantile(0.01)
        upper_bound = winsorized_df[col].quantile(0.99)
        winsorized_df[col] = winsorized_df[col].clip(
            lower=lower_bound, upper=upper_bound)

    desc_win = winsorized_df.describe()
    print(desc_win.to_string(
        float_format=lambda x: f'{x:>.2f}', justify='right'))

    # ================ 策略回測 ================
    import numpy as np
    
    # 辨識欄位：搜尋資料中帶有「代碼」與「收盤」的欄位作為計算基準
    code_col = next((col for col in winsorized_df.columns if '代碼' in col), None)
    close_col = next((col for col in winsorized_df.columns if '收盤' in col), None)
    date_col = '年月日' if '年月日' in winsorized_df.columns else None

    if code_col and close_col and date_col:
        print("\n================ 策略回測 ================ ")
        # 確保資料依據代碼與日期排序
        winsorized_df = winsorized_df.sort_values(by=[code_col, date_col]).reset_index(drop=True)
        
        # 計算指標：針對每檔股票分別計算 8 天指數移動平均 (8EMA) 與 21 天指數移動平均 (21EMA)
        winsorized_df['8EMA'] = winsorized_df.groupby(code_col)[close_col].transform(lambda x: x.ewm(span=8, adjust=False).mean())
        winsorized_df['21EMA'] = winsorized_df.groupby(code_col)[close_col].transform(lambda x: x.ewm(span=21, adjust=False).mean())
        
        # 產生訊號：
        # 若 8EMA > (21EMA * 1.01)，當天訊號為 1（買進或持有），否則為 0
        # 若 21EMA > (8EMA * 1.01)，當天訊號為 -1（賣出或持有），否則為 0
        winsorized_df['訊號'] = 0
        cond_buy = winsorized_df['8EMA'] > (winsorized_df['21EMA'] * 1.01)
        cond_sell = winsorized_df['21EMA'] > (winsorized_df['8EMA'] * 1.01)
        
        winsorized_df.loc[cond_buy, '訊號'] = 1
        winsorized_df.loc[cond_sell, '訊號'] = -1
        
        # 計算策略原始報酬：使用昨天的訊號乘上今天的股票報酬率（(收盤價 - 昨收) / 昨收）
        winsorized_df['昨收'] = winsorized_df.groupby(code_col)[close_col].shift(1)
        winsorized_df['股票報酬率'] = (winsorized_df[close_col] - winsorized_df['昨收']) / winsorized_df['昨收']
        winsorized_df['昨日訊號'] = winsorized_df.groupby(code_col)['訊號'].shift(1)
        winsorized_df['策略報酬'] = winsorized_df['昨日訊號'] * winsorized_df['股票報酬率']
        
        # 取得年份以供篩選
        winsorized_df['年份'] = pd.to_datetime(winsorized_df[date_col]).dt.year
        
        # ================ 訊號分佈情形與績效 ================
        print("\n--- 全樣本訊號分佈情形與績效 ---")
        sig_dist_results = []
        for s_val, s_label in [(1, '買進 (1)'), (-1, '賣出 (-1)'), (0, '無訊號 (0)')]:
            mask = winsorized_df['昨日訊號'] == s_val
            sr = winsorized_df.loc[mask, '策略報酬'].dropna()
            n_samples = len(sr)
            if n_samples > 0:
                mean_return = sr.mean()
                std_return = sr.std(ddof=1)
                t_stat = mean_return / (std_return / np.sqrt(n_samples)) if std_return != 0 else np.nan
            else:
                mean_return, std_return, t_stat = np.nan, np.nan, np.nan
            
            sig_dist_results.append({
                '訊號類型': s_label,
                '總樣本數 (N)': n_samples,
                '平均日報酬 (Mean)': mean_return,
                '標準差 (Std)': std_return,
                't 統計量': t_stat
            })
        print(pd.DataFrame(sig_dist_results).to_string(index=False, float_format=lambda x: f'{x:>.6f}' if pd.notnull(x) else 'NaN'))
        
        periods = {
            '2015~2019年': (2015, 2019),
            '2020~2026年': (2020, 2026)
        }
        
        signal_types = {
            '只看買進訊號': [1],
            '只看賣出訊號': [-1],
            '合併買進賣出': [1, -1]
        }
        
        results = []
        for p_name, (start_year, end_year) in periods.items():
            period_mask = (winsorized_df['年份'] >= start_year) & (winsorized_df['年份'] <= end_year)
            period_df = winsorized_df[period_mask]
            
            for s_name, s_vals in signal_types.items():
                signal_mask = period_df['昨日訊號'].isin(s_vals)
                strategy_returns = period_df.loc[signal_mask, '策略報酬'].dropna()
                
                if not strategy_returns.empty:
                    mean_return = strategy_returns.mean()
                    std_return = strategy_returns.std(ddof=1)
                    n_samples = len(strategy_returns)
                    t_stat = mean_return / (std_return / np.sqrt(n_samples)) if std_return != 0 else np.nan
                else:
                    mean_return, std_return, n_samples, t_stat = np.nan, np.nan, 0, np.nan
                
                results.append({
                    '期間': p_name,
                    '訊號類型': s_name,
                    '總樣本數 (N)': n_samples,
                    '平均日報酬 (Mean)': mean_return,
                    '標準差 (Std)': std_return,
                    't 統計量': t_stat
                })
                
        results_df = pd.DataFrame(results)
        print("\n--- 期間策略回測結果 ---")
        print(results_df.to_string(index=False, float_format=lambda x: f'{x:>.6f}' if pd.notnull(x) else 'NaN'))
        
        # 各年度合併買進賣出分析結果
        yearly_results = []
        unique_years = sorted(winsorized_df['年份'].dropna().unique())
        
        for year in unique_years:
            year_mask = winsorized_df['年份'] == year
            year_df = winsorized_df[year_mask]
            
            signal_mask = year_df['昨日訊號'].isin([1, -1])
            strategy_returns = year_df.loc[signal_mask, '策略報酬'].dropna()
            
            if not strategy_returns.empty:
                mean_return = strategy_returns.mean()
                std_return = strategy_returns.std(ddof=1)
                n_samples = len(strategy_returns)
                t_stat = mean_return / (std_return / np.sqrt(n_samples)) if std_return != 0 else np.nan
            else:
                mean_return, std_return, n_samples, t_stat = np.nan, np.nan, 0, np.nan
                
            yearly_results.append({
                '年度': int(year),
                '訊號類型': '合併買進賣出',
                '總樣本數 (N)': n_samples,
                '平均日報酬 (Mean)': mean_return,
                '標準差 (Std)': std_return,
                't 統計量': t_stat
            })
            
        yearly_results_df = pd.DataFrame(yearly_results)
        print("\n--- 各年度合併買進賣出分析結果 ---")
        print(yearly_results_df.to_string(index=False, float_format=lambda x: f'{x:>.6f}' if pd.notnull(x) else 'NaN'))

        # ================ 每日投組表現與風險調整後績效評估 ================
        print("\n================ 每日投組表現與風險調整後績效評估 ================")
        
        # 建立每日投組表現：使用 groupby('年月日') 將個別股票的報酬率加總平均
        # 我們只看有訊號的期間，因此過濾出每日的資料
        daily_portfolio = winsorized_df.groupby(date_col).agg(
            大盤報酬率=('股票報酬率', 'mean'),
            策略報酬率=('策略報酬', 'mean')
        ).dropna()
        
        if not daily_portfolio.empty and len(daily_portfolio) > 1:
            x = daily_portfolio['大盤報酬率'].values
            y = daily_portfolio['策略報酬率'].values
            
            # 計算 Alpha 與 Beta：線性回歸方程式 (y = beta * x + alpha)
            beta, alpha = np.polyfit(x, y, 1)
            
            # 計算年化 Alpha：每日 Alpha 乘以 252
            annualized_alpha = alpha * 252
            
            # 模型解釋力評估：計算關聯係數並將其平方取得 R-squared
            r_squared = np.corrcoef(x, y)[0, 1] ** 2
            
            print(f"總交易日數: {len(daily_portfolio)}")
            print(f"Beta (市場敏感度): {beta:.6f}")
            print(f"Alpha (每日絕對表現): {alpha:.6f}")
            print(f"年化 Alpha: {annualized_alpha:.6f}")
            print(f"R-squared (模型解釋力): {r_squared:.4f}")
        else:
            print("\n有效交易日數量不足，無法進行回歸分析計算 Alpha 與 Beta。")

    else:
        print("\n無法找到包含「代碼」、「收盤」或「年月日」的欄位，無法進行策略計算。")

