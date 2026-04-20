import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self):
        """
        初始化预处理模块，设定时间基准。
        主轴默认为 15 分钟（单日 96 点）。
        """
        self.points_per_day = 96
        self.segments = {
            'T1': (0, 24),   # 0-23 (含头不含尾)
            'T2': (24, 40),  # 24-39
            'T3': (40, 64),  # 40-63
            'T4': (64, 84),  # 64-83
            'T5': (84, 96)   # 84-95
        }

    def align_daily_data(self, df_daily, date_col='date'):
        """
        将日单点数据按天广播填充至当日的 96 个时段。
        适用于：断面约束、检修容量、备用需求等预测数据 (如序号 4, 6, 7, 8, 9, 12)
        :param df_daily: 包含单日数据的 DataFrame，需包含日期列和要广播的特征列。
        :return: 扩展到 96 点的 DataFrame
        """
        df_daily[date_col] = pd.to_datetime(df_daily[date_col])
        
        # 构造96点时间轴
        dates = df_daily[date_col].unique()
        time_index = []
        for d in dates:
            time_index.extend(pd.date_range(start=d, periods=self.points_per_day, freq='15T'))
            
        df_96 = pd.DataFrame({'timestamp': time_index})
        df_96[date_col] = df_96['timestamp'].dt.normalize()
        
        # 通过日期合并数据，实现广播
        df_aligned = pd.merge(df_96, df_daily, on=date_col, how='left')
        return df_aligned

    def generate_price_limits(self, df):
        """
        限价特征：根据时段自动生成 price_min 和 price_max
        T3、T4 为 10-15元，其余为 5-10元。
        需先假设数据按一天 96 点顺序排列，或者通过 'point_index' 推断时段。
        """
        if 'point_index' not in df.columns:
            # 假设 df 里有 timestamp 可以提取点位索引 (0-95)
            df['point_index'] = (df['timestamp'].dt.hour * 4 + df['timestamp'].dt.minute // 15)
            
        def get_limits(point):
            if self.segments['T3'][0] <= point < self.segments['T3'][1] or \
               self.segments['T4'][0] <= point < self.segments['T4'][1]:
                return pd.Series([10.0, 15.0])
            else:
                return pd.Series([5.0, 10.0])
                
        df[['price_min', 'price_max']] = df['point_index'].apply(get_limits)
        return df

    def calculate_net_load(self, df, load_col='Load_Predict', re_col='RE_Predict'):
        """
        净负荷计算：结合省内负荷预测和新能源出力预测
        Net_Load_Predict = Load_Predict - RE_Predict
        """
        if load_col in df.columns and re_col in df.columns:
            df['Net_Load_Predict'] = df[load_col] - df[re_col]
        else:
            raise KeyError(f'Columns "{load_col}" and/or "{re_col}" not found for net load calculation.')
            
        return df

    def calculate_deviation_feature(self, df, daily_predict_col, prev_actual_col):
        """
        偏差特征：计算前日运行实际值与当日预测值在同时间点的残差，作为模型修正因子。
        """
        if daily_predict_col in df.columns and prev_actual_col in df.columns:
            df['predict_residual'] = df[prev_actual_col] - df[daily_predict_col]
        else:
            raise KeyError('Missing required columns for deviation calculation.')
            
        return df

    def process_all(self, df_96_base, df_daily_factors):
        """
        整体数据处理流水线示例。
        """
        # 1. 对齐日单点数据
        df_aligned = self.align_daily_data(df_daily_factors)
        
        # 将日维度对齐的数据与 96 点主数据骨架合并
        df_final = pd.merge(df_96_base, df_aligned, on='timestamp', how='left')
        
        # 2. 生成限价特征
        df_final = self.generate_price_limits(df_final)
        
        # 3. 净负荷计算 (假设字段名已匹配)
        # df_final = self.calculate_net_load(df_final, 'load_predict', 're_predict')
        
        # 4. 偏差特征 (假设字段名已匹配)
        # df_final = self.calculate_deviation_feature(df_final, 'load_predict', 'load_actual_prev')
        
        return df_final

if __name__ == '__main__':
    # 简单的运行测试
    preprocessor = DataPreprocessor()
    print("Preprocessor initialized successfully.")
