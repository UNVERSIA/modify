import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from carbon_calculator import CarbonCalculator


class DataSimulator:
    def __init__(self):
        self.start_date = datetime(2018, 1, 1)
        self.end_date = datetime(2024, 12, 31)

    def _create_monthly_data(self, daily_df):
        """将日度数据聚合为月度数据 - 科学处理量级问题"""
        df = daily_df.copy()
        df['日期'] = pd.to_datetime(df['日期'])

        # 设置日期为索引
        df.set_index('日期', inplace=True)

        # 按月聚合 - 修正甲烷排放计算逻辑
        monthly_df = df.resample('M').agg({
            '处理水量(m³)': 'sum',  # 月总处理水量
            '电耗(kWh)': 'sum',  # 月总电耗
            'PAC投加量(kg)': 'sum',  # 月总PAC投加量
            'PAM投加量(kg)': 'sum',  # 月总PAM投加量
            '次氯酸钠投加量(kg)': 'sum',  # 月总次氯酸钠投加量
            '进水COD(mg/L)': 'mean',  # 平均浓度
            '出水COD(mg/L)': 'mean',  # 平均浓度
            '进水TN(mg/L)': 'mean',  # 平均浓度
            '出水TN(mg/L)': 'mean',  # 平均浓度
            'total_CO2eq': 'sum',  # 月总碳排放 - 关键修改：使用sum而非mean
            '自来水(m³/d)': 'sum',
            '脱水污泥外运量(80%)': 'sum'
        }).reset_index()

        # 数据验证和调试信息
        print(f"月度数据生成完成:")
        print(
            f"  - 月度甲烷排放范围: {monthly_df['total_CO2eq'].min():.1f} - {monthly_df['total_CO2eq'].max():.1f} kgCO2eq/月")
        print(f"  - 月度甲烷排放均值: {monthly_df['total_CO2eq'].mean():.1f} kgCO2eq/月")

        # 添加年月标识列
        monthly_df['年月'] = monthly_df['日期'].dt.strftime('%Y年%m月')

        return monthly_df

        # 为了保持与实际运营的一致性，将日均值转换为月度标准值
        # 月度数据应该代表该月的平均日排放量，而不是累积排放量
        scaling_factors = {
            '处理水量(m³)': 1.0,  # 保持日均值
            '电耗(kWh)': 1.0,  # 保持日均值
            'PAC投加量(kg)': 1.0,  # 保持日均值
            'PAM投加量(kg)': 1.0,  # 保持日均值
            '次氯酸钠投加量(kg)': 1.0,  # 保持日均值
            'total_CO2eq': 1.0,  # 关键：保持日均碳排放，不乘以天数
            '自来水(m³/d)': 1.0,
            '脱水污泥外运量(80%)': 1.0
        }

        # 应用缩放因子
        for col, factor in scaling_factors.items():
            if col in monthly_df.columns:
                monthly_df[col] = monthly_df[col] * factor

        # 添加年月标识列
        monthly_df['年月'] = monthly_df['日期'].dt.strftime('%Y年%m月')

        # 添加数据验证
        print(f"月度数据生成完成:")
        print(
            f"  - 月度甲烷排放范围: {monthly_df['total_CO2eq'].min():.1f} - {monthly_df['total_CO2eq'].max():.1f} kgCO2eq/日均")
        print(f"  - 月度甲烷排放均值: {monthly_df['total_CO2eq'].mean():.1f} kgCO2eq/日均")

        return monthly_df

    def generate_seasonal_pattern(self, length, amplitude, phase=0):
        """生成季节性模式"""
        x = np.arange(length)
        seasonal = amplitude * np.sin(2 * np.pi * x / 365 + phase)
        return seasonal

    def generate_trend(self, length, slope):
        """生成趋势成分"""
        return slope * np.arange(length)

    def generate_noise(self, length, scale):
        """生成噪声成分"""
        return np.random.normal(0, scale, length)

    def generate_water_flow(self, length):
        """生成处理水量数据"""
        base = 10000  # 基础水量
        seasonal = self.generate_seasonal_pattern(length, 2000, 0)
        trend = self.generate_trend(length, 0.5)  # 缓慢上升趋势
        noise = self.generate_noise(length, 300)
        return base + seasonal + trend + noise

    def generate_energy_consumption(self, water_flow, length):
        """生成能耗数据（与水量相关）"""
        base_ratio = 0.3  # 基础能耗系数 kWh/m³
        seasonal_var = self.generate_seasonal_pattern(length, 0.05, np.pi / 2)
        noise = self.generate_noise(length, 0.02)
        ratios = base_ratio + seasonal_var + noise
        return water_flow * ratios

    def generate_chemical_usage(self, water_flow, length):
        """生成药剂使用量数据"""
        # PAC投加量 (与水量和季节相关)
        pac_base = 0.02  # kg/m³
        pac_seasonal = self.generate_seasonal_pattern(length, 0.005, np.pi)
        pac_ratio = pac_base + pac_seasonal + self.generate_noise(length, 0.002)
        pac_usage = water_flow * pac_ratio

        # PAM投加量
        pam_base = 0.005  # kg/m³
        pam_ratio = pam_base + self.generate_noise(length, 0.001)
        pam_usage = water_flow * pam_ratio

        # 次氯酸钠投加量
        naclo_base = 0.01  # kg/m³
        naclo_ratio = naclo_base + self.generate_seasonal_pattern(length, 0.002, np.pi / 4)
        naclo_usage = water_flow * naclo_ratio

        return pac_usage, pam_usage, naclo_usage

    def generate_water_quality(self, length):
        """生成水质数据"""
        # 进水COD - 有季节性变化和缓慢改善趋势
        cod_in_base = 250
        cod_in_seasonal = self.generate_seasonal_pattern(length, 30, np.pi / 3)
        cod_in_trend = self.generate_trend(length, -0.05)  # 逐年改善
        cod_in_noise = self.generate_noise(length, 10)
        cod_in = cod_in_base + cod_in_seasonal + cod_in_trend + cod_in_noise

        # 出水COD - 处理效果逐年改善
        removal_efficiency = 0.85 + self.generate_trend(length, 0.001)  # 效率逐年提高
        cod_out = cod_in * (1 - removal_efficiency) + self.generate_noise(length, 5)

        # 进水TN
        tn_in_base = 40
        tn_in_seasonal = self.generate_seasonal_pattern(length, 8, np.pi / 2)
        tn_in_trend = self.generate_trend(length, -0.03)
        tn_in_noise = self.generate_noise(length, 3)
        tn_in = tn_in_base + tn_in_seasonal + tn_in_trend + tn_in_noise

        # 出水TN
        tn_removal = 0.75 + self.generate_trend(length, 0.002)
        tn_out = tn_in * (1 - tn_removal) + self.generate_noise(length, 1.5)

        return cod_in, cod_out, tn_in, tn_out

    def generate_simulated_data(self, save_path="data/simulated_data.csv"):
        """生成完整的模拟数据集"""
        # 修改日期范围：2018-2024年
        self.start_date = datetime(2018, 1, 1)
        self.end_date = datetime(2024, 12, 31)

        date_range = pd.date_range(self.start_date, self.end_date)
        length = len(date_range)

        # 生成各指标数据
        water_flow = self.generate_water_flow(length)
        energy_consumption = self.generate_energy_consumption(water_flow, length)
        pac_usage, pam_usage, naclo_usage = self.generate_chemical_usage(water_flow, length)
        cod_in, cod_out, tn_in, tn_out = self.generate_water_quality(length)

        # 构建DataFrame - 确保包含LSTM预测器所需的所有列
        data = {
            "日期": date_range,
            "处理水量(m³)": np.round(water_flow),
            "电耗(kWh)": np.round(energy_consumption),
            "PAC投加量(kg)": np.round(pac_usage),
            "PAM投加量(kg)": np.round(pam_usage),
            "次氯酸钠投加量(kg)": np.round(naclo_usage),
            "进水COD(mg/L)": np.round(cod_in, 1),
            "出水COD(mg/L)": np.round(cod_out, 1),
            "进水TN(mg/L)": np.round(tn_in, 1),
            "出水TN(mg/L)": np.round(tn_out, 1),
            # 添加一些可能用到的其他列
            "自来水(m³/d)": np.round(water_flow * 0.05),  # 假设自来水用量为处理水量的5%
            "脱水污泥外运量(80%)": np.round(water_flow * 0.001)  # 假设污泥产量为处理水量的0.1%
        }

        df = pd.DataFrame(data)

        # 确保没有NaN值
        df = df.fillna(0)

        # 确保所有数值都是正数
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].abs()

        # 计算甲烷排放数据（确保包含total_CO2eq列）
        try:
            calculator = CarbonCalculator()
            df_with_emissions = calculator.calculate_direct_emissions(df)
            df_with_emissions = calculator.calculate_indirect_emissions(df_with_emissions)
            df_with_emissions = calculator.calculate_unit_emissions(df_with_emissions)
            df = df_with_emissions
        except Exception as e:
            print(f"计算甲烷排放数据时出错: {e}")
            # 如果计算失败，添加一个默认的total_CO2eq列
            df['total_CO2eq'] = df['电耗(kWh)'] * 0.5 + df['处理水量(m³)'] * 0.1

        # 新增：创建月度聚合数据
        df_monthly = self._create_monthly_data(df)

        # 保存到文件
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False, encoding='utf-8')
        df_monthly.to_csv(save_path.replace('.csv', '_monthly.csv'), index=False, encoding='utf-8')

        print(f"模拟数据已生成并保存到 {save_path}，共 {len(df)} 条记录")
        print(f"月度数据已生成并保存到 {save_path.replace('.csv', '_monthly.csv')}，共 {len(df_monthly)} 条记录")

        return df


# 使用示例
if __name__ == "__main__":
    simulator = DataSimulator()
    simulated_data = simulator.generate_simulated_data()
