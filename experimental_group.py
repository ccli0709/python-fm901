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

    # 進行 8EMA / 21EMA 策略回測 (基於回踩結構的增量價值測試)
    print("\n================ 8EMA / 21EMA 策略回測 (回踩結構增量價值測試) ================")
    import numpy as np

    # 尋找代碼、收盤價與最低價欄位
    code_col = next(
        (col for col in winsorized_df.columns if '代碼' in col), winsorized_df.columns[0])
    close_col = next(
        (col for col in winsorized_df.columns if '收盤' in col), None)
    low_col = next(
        (col for col in winsorized_df.columns if '最低' in col or 'Low' in col or 'low' in col), close_col) # 若無最低價則預設用收盤價替代

    if close_col:
        # 確保資料依據股票代碼與日期排序
        winsorized_df = winsorized_df.sort_values(by=[code_col, '年月日'])

        # 計算 8EMA 與 21EMA
        winsorized_df['8EMA'] = winsorized_df.groupby(code_col)[close_col].transform(
            lambda x: x.ewm(span=8, adjust=False).mean())
        winsorized_df['21EMA'] = winsorized_df.groupby(code_col)[close_col].transform(
            lambda x: x.ewm(span=21, adjust=False).mean())

        # 計算前一天的數值，用於判斷交叉與回檔狀態
        winsorized_df['Prev_8EMA'] = winsorized_df.groupby(code_col)['8EMA'].shift(1)
        winsorized_df['Prev_21EMA'] = winsorized_df.groupby(code_col)['21EMA'].shift(1)
        winsorized_df['Prev_Close'] = winsorized_df.groupby(code_col)[close_col].shift(1)

        # X1 (單純交叉)：8 EMA 向上穿越 21 EMA 發出的買入訊號
        winsorized_df['X1_Signal'] = (winsorized_df['8EMA'] > winsorized_df['21EMA']) & \
                                     (winsorized_df['Prev_8EMA'] <= winsorized_df['Prev_21EMA'])

        # X2 (回踩確認)：在 8EMA > 21EMA 期間，價格回檔 (收盤下跌) 但最低價未跌破 21 EMA
        winsorized_df['X2_Signal'] = (winsorized_df['8EMA'] > winsorized_df['21EMA']) & \
                                     (winsorized_df[close_col] < winsorized_df['Prev_Close']) & \
                                     (winsorized_df[low_col] >= winsorized_df['21EMA']) & \
                                     (~winsorized_df['X1_Signal']) # 排除 X1 發生的當天

        # X3 (結構確立)：在 X2 發生後的隔天，收盤價上漲且成功站回(大於等於) 21 EMA
        winsorized_df['Prev_X2'] = winsorized_df.groupby(code_col)['X2_Signal'].shift(1)
        winsorized_df['X3_Signal'] = (winsorized_df['8EMA'] > winsorized_df['21EMA']) & \
                                     (winsorized_df['Prev_X2'] == True) & \
                                     (winsorized_df[close_col] > winsorized_df['Prev_Close']) & \
                                     (winsorized_df[close_col] >= winsorized_df['21EMA'])

        # 設定未來持倉 n 日 (此處預設為 10 日，您可以根據需求修改)
        hold_days = 10
        
        # 計算未來 n 日的對數報酬率: ln(Close_t+n / Close_t)
        winsorized_df[f'Future_{hold_days}d_Close'] = winsorized_df.groupby(code_col)[close_col].shift(-hold_days)
        winsorized_df['Future_Log_Return'] = np.log(winsorized_df[f'Future_{hold_days}d_Close'] / winsorized_df[close_col])

        # 建立評估訊號績效的函數
        def evaluate_signal(signal_col, signal_name):
            # 取出該訊號發生且未來報酬率不為空的樣本
            samples = winsorized_df[winsorized_df[signal_col] == True]['Future_Log_Return'].dropna()
            n_samples = len(samples)
            
            if n_samples > 0:
                y1_mean = samples.mean() # Y1: 條件平均報酬率 (對數)
                y2_vol = samples.std()   # Y2: 條件波動率
                # Y3: 夏普比率 (假設無風險利率為 0。此處為持有期間夏普，若需年化可乘上 sqrt(252/n))
                y3_sharpe = (y1_mean / y2_vol) if y2_vol != 0 else np.nan
                
                print(f"\n--- {signal_name} ---")
                print(f"樣本數 (筆): {n_samples}")
                print(f"Y1 (條件平均對數報酬率, {hold_days}日): {y1_mean:.6f}")
                print(f"Y2 (條件波動率 / 標準差): {y2_vol:.6f}")
                print(f"Y3 (夏普比率): {y3_sharpe:.4f}")
            else:
                print(f"\n--- {signal_name} ---")
                print("無有效樣本。")

        print(f"\n衡量績效與風險 (設定持倉 n = {hold_days} 日)")
        evaluate_signal('X1_Signal', 'X1 (單純交叉)')
        evaluate_signal('X2_Signal', 'X2 (回踩確認)')
        evaluate_signal('X3_Signal', 'X3 (結構確立)')

    else:
        print("找不到收盤價相關欄位，無法進行策略計算。")

else:
    print("沒有讀取到任何資料。")
