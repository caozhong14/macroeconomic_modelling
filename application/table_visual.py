import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class table_visual:
    def __init__(self, loader_dir, country_region_map_path, output_dir,yearstart, year,thousand_format):
        self.file_path = loader_dir
        self.country_region_map_path = country_region_map_path
        self.output_dir = output_dir
        self.yearstart = yearstart
        self.processed_df = None
        self.raw_df = None
        self.year = year
        self.country_region_map = self._load_country_region_map() 
        self.decimal1 = 0
        self.decimal2 = 3 
        self.thousand_format=thousand_format
       
        os.makedirs(self.output_dir, exist_ok=True)
           
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
       
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        log_file_path = os.path.join(self.output_dir, 'table_logfile.log')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _load_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.raw_df = pd.read_csv(f)
        self._validate_data()
        self.raw_df = self.raw_df[self.raw_df['year'] == self.year]
        self.raw_df = self.raw_df.drop(columns=['year'])
        return self.raw_df

    def _load_country_region_map(self):
        """加载国家代码到国家名称的映射"""
        try:
            country_region_df = pd.read_csv(self.country_region_map_path, encoding='gbk')
            country_region_map = dict(zip(country_region_df['Country Code'], country_region_df['country']))
            region_map = country_region_df.set_index('country')['Region'].to_dict()
            income_group_map = country_region_df.set_index('country')['Income group'].to_dict()
            return country_region_map, region_map, income_group_map
        except Exception as e:
            raise ValueError(f"无法加载国家区域映射文件: {e}")

    def _validate_data(self):
        """验证数据格式是否符合要求"""
        if 'year' not in self.raw_df.columns:
            raise ValueError("year")
        if 'country_codes' not in self.raw_df.columns:
            raise ValueError("country_codes")

    def _format_number(self, value, decimal):
        """格式化数字"""
        if decimal==0:
            return f"{value:.0f}"
        else:
            return f"{value:.{decimal}f}"

    def process_three_columns(self):
       
       
       
        prefixes=['GDPloss','GDPShare','GDPper']
       
        self.processed_df = pd.DataFrame()
        self.processed_df['country'] = self.raw_df['country_codes'].map(self.country_region_map[0]) 
        self.processed_df['region'] = self.processed_df['country'].map(self.country_region_map[1])

       
        for prefix in sorted(prefixes):
            if prefix== 'GDPShare':
                decimal = self.decimal1
            else:
                decimal = self.decimal1
            val_col = f"{prefix}_val"
            upper_col = f"{prefix}_upper"
            lower_col = f"{prefix}_lower"
           
            if all(col in self.raw_df.columns for col in [val_col, upper_col, lower_col]):
               
                self.processed_df[prefix] = self.raw_df.apply(
                    lambda row: (
                        f"{self._format_number(row[val_col], decimal)}"
                        f"({self._format_number(row[lower_col], decimal)}-"
                        f"{self._format_number(row[upper_col], decimal)})"
                    ), axis=1
                )
            else:
                print(f"警告: {prefix} 的完整数据列不存在，已跳过")

        return self.processed_df

    def group_and_sort_data(self):
        """按区域分组并按国家名称排序"""
        if self.processed_df is None:
            print("警告: 没有处理后的数据可供分组和排序")
            return None

       
        grouped = self.processed_df.groupby('region', group_keys=True).apply(lambda x: x.sort_values(by='country'))

       
        grouped = grouped.reset_index(drop=True)

        return grouped

    def save_grouped_data(self, grouped_data, filename):
        """保存分组排序后的数据"""
        if grouped_data is not None:
            output_path = os.path.join(self.output_dir, filename)
            grouped_data.to_csv(output_path, index=False)
            print(f"分组排序后的数据已保存到: {output_path}")
            
           
            base_name = filename.split('.')[0]
            if '_format' in base_name:
                base_name = base_name.replace('_format', '')
            
           
            raw_filename = f"{base_name}_raw.csv"
            raw_output_path = os.path.join(self.output_dir, raw_filename)
            
           
           
           
           
            
           
            if "table1" in base_name or "table2" in base_name or "table3" in base_name or "table4" in base_name:
               
                raw_df = self.raw_df.copy()
                raw_df['country'] = raw_df['country_codes'].map(self.country_region_map[0])
                raw_df['region'] = raw_df['country'].map(self.country_region_map[1])
                
               
                columns_to_keep = [
                    'country_codes', 'country', 'region',
                    'GDPloss_val', 'GDPloss_lower', 'GDPloss_upper',
                    'GDPShare_val', 'GDPShare_lower', 'GDPShare_upper',
                    'GDPper_val', 'GDPper_lower', 'GDPper_upper'
                ]
                raw_df = raw_df[columns_to_keep]
                
               
                raw_df = raw_df.rename(columns={
                    'GDPloss_val': 'Economic_cost_val',
                    'GDPloss_lower': 'Economic_cost_lower',
                    'GDPloss_upper': 'Economic_cost_upper',
                    'GDPShare_val': 'GDP_share_val',
                    'GDPShare_lower': 'GDP_share_lower',
                    'GDPShare_upper': 'GDP_share_upper',
                    'GDPper_val': 'Per_capita_cost_val',
                    'GDPper_lower': 'Per_capita_cost_lower',
                    'GDPper_upper': 'Per_capita_cost_upper'
                })
                
                raw_df.to_csv(raw_output_path, index=False)
                
            
            print(f"原始数据已保存到: {raw_output_path}")
        else:
            print("警告: 没有分组排序后的数据可供保存")

    def _table_discount_visual(self, mode):
        self.raw_df = self._load_data()
       
        if mode == "exp1_baseline":
            self.raw_df = self.raw_df[self.raw_df['task'] == "exp1_baseline"]
        elif mode == "exp2_varparams":
            self.raw_df = self.raw_df[self.raw_df['task'] == "exp2_varparams"]
        elif mode == "exp3_vardiscount2":
            self.raw_df = self.raw_df[self.raw_df['task'] == "exp3_vardiscount2"]
        elif mode == "exp4_vardiscount0":
            self.raw_df = self.raw_df[self.raw_df['task'] == "exp4_vardiscount0"]

       
        columns_to_keep = [
            'country_codes', 'GDPloss_val', 'GDPloss_lower', 'GDPloss_upper',
            'GDPShare_val', 'GDPShare_lower', 'GDPShare_upper',
            'GDPper_val', 'GDPper_lower', 'GDPper_upper'
        ]
        self.raw_df = self.raw_df[columns_to_keep]

       
        gdp_loss_columns = [col for col in self.raw_df.columns if col.startswith('GDPloss')]
        for col in gdp_loss_columns:
            self.raw_df[col] = self.raw_df[col] / 1e6

       
        gdp_share_columns = [col for col in self.raw_df.columns if col.startswith('GDPShare')]
        for col in gdp_share_columns:
            scale_factor = 1e2 * (1e3 if self.thousand_format else 1)
            self.raw_df[col] = self.raw_df[col] * scale_factor
           
            if self.thousand_format:
                self.raw_df[col] = self.raw_df[col].round(0)
           

       
        self.processed_df = self.process_three_columns()

       
        new_df = pd.DataFrame()
        regions = sorted(self.processed_df['region'].unique())
        combined_list = []
        for region in regions:
            combined_list.append(region)
            countries = sorted(self.processed_df[self.processed_df['region'] == region]['country'].tolist())
            combined_list.extend(countries)
        new_df['Country/Region'] = combined_list

       
        other_cols = [col for col in self.processed_df.columns if col not in ['country', 'region']]
        for col in other_cols:
            values = []
            for item in combined_list:
                if item in regions:
                    values.append('')
                else:
                    match = self.processed_df[self.processed_df['country'] == item][col]
                    values.append(match.iloc[0] if not match.empty else '')
            new_df[col] = values

       
        share_name = (
            'Proportion of total GDP (× 10⁻³ %), 2025–2050'
            if self.thousand_format else
            'Proportion of total GDP (%), 2025–2050'
        )
        new_df = new_df.rename(columns={
            'Country/Region': '',
            'GDPloss': 'Economic cost, 2024 U.S.$ million',
            'GDPShare': share_name,
            'GDPper': 'Per capita cost, 2024 U.S.$'
        })

       
        new_order = [
            '',
            'Economic cost, 2024 U.S.$ million',
            share_name,
            'Per capita cost, 2024 U.S.$'
        ]
        new_df_order = new_df[new_order]

       
        filename_map = {
            'exp1_baseline': 'table1.csv',
            'exp2_varparams': 'table2.csv',
            'exp3_vardiscount2': 'table3.csv',
            'exp4_vardiscount0': 'table4.csv'
        }
        log_map = {
            'exp1_baseline': "table1:Total economic cost, cost as a percentage of GDP in 2025–50, and per capita cost, by country and region",
            'exp2_varparams': "table2:Economic cost, the cost as a percentage of GDP in 2025–50, and per capita cost, by country and region with different parameters",
            'exp3_vardiscount2': "table3:Economic cost, the cost as a percentage of GDP in 2025–50, and per capita cost, by country and region with a discount rate of 2%",
            'exp4_vardiscount0': "table4:Economic cost, the cost as a percentage of GDP in 2025–50, and per capita cost, by country and region with a discount rate of 0%"
        }
        if mode in filename_map:
            self.logger.info(log_map[mode])
            self.save_grouped_data(new_df_order, filename=filename_map[mode])


    def _table_discount_visual_full_format(self, mode):
        self.raw_df = self._load_data()
       
        if mode == "exp1_baseline":
            self.raw_df = self.raw_df[self.raw_df['task'] == "exp1_baseline"]
        elif mode == "exp2_varparams":
            self.raw_df = self.raw_df[self.raw_df['task'] == "exp2_varparams"]
        elif mode == "exp3_vardiscount2":
            self.raw_df = self.raw_df[self.raw_df['task'] == "exp3_vardiscount2"]
        elif mode == "exp4_vardiscount0":
            self.raw_df = self.raw_df[self.raw_df['task'] == "exp4_vardiscount0"]

       
        columns_to_keep = [
            'country_codes', 'GDPloss_val', 'GDPloss_lower', 'GDPloss_upper',
            'GDPShare_val', 'GDPShare_lower', 'GDPShare_upper',
            'GDPper_val', 'GDPper_lower', 'GDPper_upper'
        ]
        self.raw_df = self.raw_df[columns_to_keep]

       
        gdp_loss_columns = [col for col in self.raw_df.columns if col.startswith('GDPloss')]
        for col in gdp_loss_columns:
            self.raw_df[col] = self.raw_df[col] / 1e6

       
        gdp_share_columns = [col for col in self.raw_df.columns if col.startswith('GDPShare')]
        for col in gdp_share_columns:
            scale_factor = 1e2 * (1e3 if self.thousand_format else 1)
            self.raw_df[col] = self.raw_df[col] * scale_factor
            if self.thousand_format:
               
                self.raw_df[col] = self.raw_df[col].round(0)

       
        self.processed_df = self.process_three_columns()
        grouped_data = self.group_and_sort_data()

       
        share_name = (
            'Percentage of total GDP in 2025–2050 (× 10⁻³ %)' if self.thousand_format
            else 'Percentage of total GDP in 2025–2050, %'
        )
        grouped_data = grouped_data[['region', 'country', 'GDPloss', 'GDPShare', 'GDPper']]
        grouped_data = grouped_data.rename(columns={
            'region': 'Region',
            'country': 'Country',
            'GDPloss': 'Economic cost, 2024 U.S.$ million',
            'GDPShare': share_name,
            'GDPper': 'Per capita cost, 2024 U.S.$'
        })

       
        new_order = [
            'Region', 'Country',
            'Economic cost, 2024 U.S.$ million',
            share_name,
            'Per capita cost, 2024 U.S.$'
        ]
        ordered_df = grouped_data[new_order]

       
        filename_map = {
            'exp1_baseline': 'table1_format.csv',
            'exp2_varparams': 'table2_format.csv',
            'exp3_vardiscount2': 'table3_format.csv',
            'exp4_vardiscount0': 'table4_format.csv'
        }
        log_map = {
            'exp1_baseline': "table1_format:Total economic cost, cost as a percentage of GDP in 2025–50, and per capita cost, by country and region",
            'exp2_varparams': "table2_format:Economic cost, the cost as a percentage of GDP in 2025–50, and per capita cost, by country and region with different parameters",
            'exp3_vardiscount2': "table3_format:Economic cost, the cost as a percentage of GDP in 2025–50, and per capita cost, by country and region with a discount rate of 2%",
            'exp4_vardiscount0': "table4_format:Economic cost, the cost as a percentage of GDP in 2025–50, and per capita cost, by country and region with a discount rate of 0%"
        }
        if mode in filename_map:
            self.logger.info(log_map[mode])
            self.save_grouped_data(ordered_df, filename=filename_map[mode])

    def _table_eco_agg_visual(self, mode):
        self.raw_df = self._load_data()
       
        if mode == "exp1_baseline":
            df_raw = self.raw_df[self.raw_df['task'] == "exp1_baseline"]
        elif mode == "exp2_varparams":
            df_raw = self.raw_df[self.raw_df['task'] == "exp2_varparams"]
        elif mode == "exp3_vardiscount2":
            df_raw = self.raw_df[self.raw_df['task'] == "exp3_vardiscount2"]
        elif mode == "exp4_vardiscount0":
            df_raw = self.raw_df[self.raw_df['task'] == "exp4_vardiscount0"]
        else:
            df_raw = self.raw_df.copy()

       
        columns_to_keep = ['country_codes', 'GDPloss_val', 'GDPloss_lower', 'GDPloss_upper',
                        'GDP', 'POP_annual']
        df_raw = df_raw[columns_to_keep]

       
        scale_factor = 1e9 
        percentage_base = 100 * (1e3 if self.thousand_format else 1)
        share_label = 'Proportion of total GDP ' + (
            '(× 10⁻³ %), 2025–2050' if self.thousand_format else '(%), 2025–2050'
        )

       
        countries = df_raw['country_codes'].map(self.country_region_map[0])
        df = pd.DataFrame({
            'Country': countries,
            'Region': countries.map(self.country_region_map[1]),
            'Income Group': countries.map(self.country_region_map[2]),
            'GDPloss_val': df_raw['GDPloss_val'],
            'GDPloss_lower': df_raw['GDPloss_lower'],
            'GDPloss_upper': df_raw['GDPloss_upper'],
            'GDP': df_raw['GDP'],
            'POP_annual': df_raw['POP_annual']
        })

       
        regions = ['East Asia and Pacific', 'Europe and Central Asia', 'Latin America and Caribbean',
                   'Middle East and North Africa', 'North America', 'South Asia', 'Sub-Saharan Africa']
        income_groups = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
       
        raw_rows = []
       
        display_rows = []

       
        def calc(name, group_df):
            val = group_df['GDPloss_val'].sum()
            lo = group_df['GDPloss_lower'].sum()
            hi = group_df['GDPloss_upper'].sum()
            total_gdp = group_df['GDP'].sum()
            p_val = val / total_gdp
            p_lo = lo / total_gdp
            p_hi = hi / total_gdp
           
            total_pop = group_df['POP_annual'].sum()
            pc_val = val / total_pop
            pc_lo = lo / total_pop
            pc_hi = hi / total_pop
           
            raw = {
                'Group': name,
                'Economic_cost_val': val,
                'Economic_cost_lower': lo,
                'Economic_cost_upper': hi,
                'GDP_share_val': p_val,
                'GDP_share_lower': p_lo,
                'GDP_share_upper': p_hi,
                'Per_capita_cost_val': pc_val,
                'Per_capita_cost_lower': pc_lo,
                'Per_capita_cost_upper': pc_hi
            }
           
            share_decimals = 0 if self.thousand_format else self.decimal2
           
            disp = {
                'Group': name,
                'Economic costs, 2024 U.S.$ billion': f"{self._format_number(val/scale_factor, self.decimal1)}(" \
                                                f"{self._format_number(lo/scale_factor, self.decimal1)}-" \
                                                f"{self._format_number(hi/scale_factor, self.decimal1)})",
                share_label: f"{self._format_number(p_val*percentage_base, share_decimals)}(" \
                            f"{self._format_number(p_lo*percentage_base, share_decimals)}-" \
                            f"{self._format_number(p_hi*percentage_base, share_decimals)})",
                'Per capita cost, 2024 U.S.$': f"{self._format_number(pc_val, self.decimal1)}(" \
                                                f"{self._format_number(pc_lo, self.decimal1)}-" \
                                                f"{self._format_number(pc_hi, self.decimal1)})"
            }
            return raw, disp

       
        display_rows.append({'Group': 'By region', 'Economic costs, 2024 U.S.$ billion': '', share_label: '', 'Per capita cost, 2024 U.S.$': ''})
        raw_rows.append({'Group': 'By region', 'Economic_cost_val': None, 'Economic_cost_lower': None, 'Economic_cost_upper': None,
                        'GDP_share_val': None, 'GDP_share_lower': None, 'GDP_share_upper': None,
                        'Per_capita_cost_val': None, 'Per_capita_cost_lower': None, 'Per_capita_cost_upper': None})
        for r in regions:
            raw, disp = calc(r, df[df['Region'] == r])
            raw_rows.append(raw)
            display_rows.append(disp)
       
        display_rows.append({'Group': 'By income group', 'Economic costs, 2024 U.S.$ billion': '', share_label: '', 'Per capita cost, 2024 U.S.$': ''})
        raw_rows.append({'Group': 'By income group', 'Economic_cost_val': None, 'Economic_cost_lower': None, 'Economic_cost_upper': None,
                        'GDP_share_val': None, 'GDP_share_lower': None, 'GDP_share_upper': None,
                        'Per_capita_cost_val': None, 'Per_capita_cost_lower': None, 'Per_capita_cost_upper': None})
        for ig in income_groups:
            raw, disp = calc(ig, df[df['Income Group'] == ig])
            raw_rows.append(raw)
            display_rows.append(disp)
       
        raw, disp = calc('Total', df)
        raw_rows.append(raw)
        display_rows.append(disp)

       
        raw_summary = pd.DataFrame(raw_rows)
        grouped_raw_filename = {
            'exp1_baseline': 'table5_raw.csv',
            'exp2_varparams': 'table6_raw.csv',
            'exp3_vardiscount2': 'table7_raw.csv',
            'exp4_vardiscount0': 'table8_raw.csv'
        }[mode]
        raw_summary.to_csv(os.path.join(self.output_dir, grouped_raw_filename), index=False)

        summary = pd.DataFrame(display_rows)
        grouped_filename = {
            'exp1_baseline': 'table5.csv',
            'exp2_varparams': 'table6.csv',
            'exp3_vardiscount2': 'table7.csv',
            'exp4_vardiscount0': 'table8.csv'
        }[mode]
        log_map = {
            'exp1_baseline': "table5:Total economic cost, cost as a percentage of GDP in 2025–50, and per capita cost, by region and SDI value quantile groupings",
            'exp2_varparams': "table6:Economic cost, the cost as a percentage of GDP in 2025–50, and per capita cost, by region and SDI value quantile groupings with different parameters",
            'exp3_vardiscount2': "table7:Economic cost, the cost as a percentage of GDP in 2025–50, and per capita cost, by region and SDI value quantile groupings with a discount rate of 2%",
            'exp4_vardiscount0': "table8:Economic cost, the cost as a percentage of GDP in 2025–50, and per capita cost, by region and SDI value quantile groupings with a discount rate of 0%"
        }
        self.logger.info(log_map[mode])
        self.save_grouped_data(summary, filename=grouped_filename)


    def _table_mor_agg_visual(self, mode):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.raw_df = pd.read_csv(f)

        million_factor = 1e6
        billion_factor = 1e9
        thousand_factor = 1e3
        hundred_factor = 1e2
        year_factor = self.year - self.yearstart + 1
        decimal=3

       
        if mode == "exp1_baseline":
            self.raw_df = self.raw_df[self.raw_df['task'] == "exp1_baseline"]
        elif mode == "exp2_varparams":
            self.raw_df = self.raw_df[self.raw_df['task'] == "exp2_varparams"]
        elif mode == "exp3_vardiscount2":
            self.raw_df = self.raw_df[self.raw_df['task'] == "exp3_vardiscount2"]
        elif mode == "exp4_vardiscount0":
            self.raw_df = self.raw_df[self.raw_df['task'] == "exp4_vardiscount0"]
       
        columns_to_keep = ['country_codes', 'GDPloss_val', 'GDP', 'POP', 'year', 'Morbidity', 'Mortality', 'DALYs']
        self.raw_df = self.raw_df[columns_to_keep]

       
        df = pd.DataFrame()
        df['Country'] = self.raw_df['country_codes'].map(self.country_region_map[0])
        df['Region'] = df['Country'].map(self.country_region_map[1])
        df['Income Group'] = df['Country'].map(self.country_region_map[2])

       
        df = pd.concat([df, self.raw_df], axis=1)
        df = df.drop(columns=['country_codes'])

       
        regions = ['East Asia and Pacific', 'Europe and Central Asia', 'Latin America and Caribbean',
                'Middle East and North Africa', 'North America', 'South Asia', 'Sub-Saharan Africa']
        income_groups = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
       
        raw_summary = pd.DataFrame(columns=['Group',
                                            'Economic_cost_val', 'Economic_cost_percent',
                                            f'Morbidity_{self.yearstart}_val', f'Morbidity_{self.yearstart}_percent',
                                            f'Morbidity_{self.year}_val', f'Morbidity_{self.year}_percent',
                                            f'Mortality_{self.yearstart}_val', f'Mortality_{self.yearstart}_percent',
                                            f'Mortality_{self.year}_val', f'Mortality_{self.year}_percent',
                                            f'DALYs_{self.yearstart}_val', f'DALYs_{self.yearstart}_percent',
                                            f'DALYs_{self.year}_val', f'DALYs_{self.year}_percent',
                                            'Annual_GDP_val', 'Annual_GDP_percent',
                                            'Annual_population_val', 'Annual_population_percent'])
       
        summary = pd.DataFrame(columns=['Group', 'Economic costs, 2024 U.S.$ billion (%)',
                                       
                                        f'Morbidity in {self.yearstart}, thousand (%)', f'Morbidity in {self.year}, thousand (%)',
                                        f'Mortality number in {self.yearstart} (%)', f'Mortality number in {self.year} (%)',
                                       
                                        f'DALYs in {self.yearstart}, thousand (%)', f'DALYs in {self.year}, thousand (%)', 
                                        f'Annual GDP 2025–50, 2024 U.S.$ billion (global %)',
                                        f'Annual population 2025–50, millions (global %)'])

       
        summary = pd.concat([summary, pd.DataFrame([{'Group': 'By region',
                                                    'Economic costs, 2024 U.S.$ billion (%)': '',
                                                   
                                                   
                                                    f'Morbidity in {self.yearstart}, thousand (%)': '', 
                                                    f'Morbidity in {self.year}, thousand (%)': '',
                                                    f'Mortality number in {self.yearstart} (%)': '',
                                                    f'Mortality number in {self.year} (%)': '',
                                                   
                                                   
                                                    f'DALYs in {self.yearstart}, thousand (%)': '', 
                                                    f'DALYs in {self.year}, thousand (%)': '',
                                                    f'Annual GDP 2025–50, 2024 U.S.$ billion (global %)': '',
                                                    f'Annual population 2025–50, millions (global %)': ''}])],
                            ignore_index=True)

       
        for region in regions:
           
            region_df = df[df['Region'] == region]
            df_yearstart = df[df['year'] == self.yearstart]
            df_year = df[df['year'] == self.year]
            region_df_yearstart = region_df[region_df['year'] == self.yearstart]
            region_df_year = region_df[region_df['year'] == self.year]

           
            GDPloss_val_total = region_df_year['GDPloss_val'].sum()
            GDPloss_val_global = df_year['GDPloss_val'].sum()
            GDPloss_percentage = (GDPloss_val_total / GDPloss_val_global) * 100
            Economic_burden_total = f"{self._format_number(GDPloss_val_total / billion_factor, 0)}({self._format_number(GDPloss_percentage, decimal)}%)"

           
            morbidities_yearstart = (region_df_yearstart['POP'] * region_df_yearstart['Morbidity']).sum()
            morbidities_yearstart_global = (df_yearstart['POP'] * df_yearstart['Morbidity']).sum()
            morbidities_yearstart_percentage = (morbidities_yearstart / morbidities_yearstart_global) * 100
            morbidities_yearstart_total = f"{self._format_number(morbidities_yearstart/thousand_factor, 0)}({self._format_number(morbidities_yearstart_percentage, decimal)}%)"

            morbidities_year = (region_df_year['POP'] * region_df_year['Morbidity']).sum()
            morbidities_year_global = (df_year['POP'] * df_year['Morbidity']).sum()
            morbidities_year_percentage = (morbidities_year / morbidities_year_global) * 100
            morbidities_year_total = f"{self._format_number(morbidities_year/thousand_factor, 0)}({self._format_number(morbidities_year_percentage, decimal)}%)"

           
            mortalities_yearstart = (region_df_yearstart['POP'] * region_df_yearstart['Mortality']).sum()
            mortalities_yearstart_global = (df_yearstart['POP'] * df_yearstart['Mortality']).sum()
            mortalities_yearstart_percentage = (mortalities_yearstart / mortalities_yearstart_global) * 100
            mortalities_yearstart_total = f"{self._format_number(mortalities_yearstart, 0)}({self._format_number(mortalities_yearstart_percentage,decimal)}%)"

            mortalities_year = (region_df_year['POP'] * region_df_year['Mortality']).sum()
            mortalities_year_global = (df_year['POP'] * df_year['Mortality']).sum()
            mortalities_year_percentage = (mortalities_year / mortalities_year_global) * 100
            mortalities_year_total = f"{self._format_number(mortalities_year, 0)}({self._format_number(mortalities_year_percentage, decimal)}%)"

           
            dalys_yearstart = (region_df_yearstart['POP'] * region_df_yearstart['DALYs']).sum()
            dalys_yearstart_global = (df_yearstart['POP'] * df_yearstart['DALYs']).sum()
            dalys_yearstart_percentage = (dalys_yearstart / dalys_yearstart_global) * 100
            dalys_yearstart_total = f"{self._format_number(dalys_yearstart/thousand_factor, 0)}({self._format_number(dalys_yearstart_percentage, decimal)}%)"

            dalys_year = (region_df_year['POP'] * region_df_year['DALYs']).sum()
            dalys_year_global = (df_year['POP'] * df_year['DALYs']).sum()
            dalys_year_percentage = (dalys_year / dalys_year_global) * 100
            dalys_year_total = f"{self._format_number(dalys_year/thousand_factor, 0)}({self._format_number(dalys_year_percentage, decimal)}%)"

           
            sfGDP_val_total = region_df_year['GDP'].sum()
            sfGDP_val_global = df_year['GDP'].sum()
            sfGDP_percentage = (sfGDP_val_total / sfGDP_val_global) * 100
            sfGDP_total = f"{self._format_number(sfGDP_val_total / year_factor / billion_factor, 0)}({self._format_number(sfGDP_percentage, decimal)}%)"

            dfPOP_val_total = region_df['POP'].sum()
            dfPOP_val_global = df['POP'].sum()
            dfPOP_percentage = (dfPOP_val_total / dfPOP_val_global) * 100
            dfPOP_total = f"{self._format_number(dfPOP_val_total / million_factor / year_factor, 0)}({self._format_number(dfPOP_percentage, decimal)}%)"

           
            summary = pd.concat([summary, pd.DataFrame([{'Group': region,
                                                        'Economic costs, 2024 U.S.$ billion (%)': Economic_burden_total,
                                                       
                                                       
                                                        f'Morbidity in {self.yearstart}, thousand (%)': morbidities_yearstart_total,
                                                        f'Morbidity in {self.year}, thousand (%)': morbidities_year_total,
                                                        f'Mortality number in {self.yearstart} (%)': mortalities_yearstart_total,
                                                        f'Mortality number in {self.year} (%)': mortalities_year_total,
                                                       
                                                       
                                                        f'DALYs in {self.yearstart}, thousand (%)': dalys_yearstart_total, 
                                                        f'DALYs in {self.year}, thousand (%)': dalys_year_total, 
                                                        f'Annual GDP 2025–50, 2024 U.S.$ billion (global %)': sfGDP_total,
                                                        f'Annual population 2025–50, millions (global %)': dfPOP_total}])],
                                ignore_index=True)
            raw_summary = pd.concat([raw_summary,pd.DataFrame([{
                'Group': region,
                'Economic_cost_val': GDPloss_val_total,
                'Economic_cost_percent': GDPloss_percentage,
                f'Morbidity_{self.yearstart}_val': morbidities_yearstart,
                f'Morbidity_{self.yearstart}_percent': morbidities_yearstart_percentage,
                f'Morbidity_{self.year}_val': morbidities_year,
                f'Morbidity_{self.year}_percent': morbidities_year_percentage,
                f'Mortality_{self.yearstart}_val': mortalities_yearstart,
                f'Mortality_{self.yearstart}_percent': mortalities_yearstart_percentage,
                f'Mortality_{self.year}_val': mortalities_year,
                f'Mortality_{self.year}_percent': mortalities_year_percentage,
                f'DALYs_{self.yearstart}_val': dalys_yearstart,
                f'DALYs_{self.yearstart}_percent': dalys_yearstart_percentage,
                f'DALYs_{self.year}_val': dalys_year,
                f'DALYs_{self.year}_percent': dalys_year_percentage,
                'Annual_GDP_val': sfGDP_val_total,
                'Annual_GDP_percent': sfGDP_percentage,
                'Annual_population_val': dfPOP_val_total,
                'Annual_population_percent': dfPOP_percentage
            }])], ignore_index=True)

       
        summary = pd.concat([summary, pd.DataFrame([{'Group': 'By income group',
                                                    'Economic costs, 2024 U.S.$ billion (%)': '',
                                                   
                                                   
                                                    f'Morbidity in {self.yearstart}, thousand (%)': '',
                                                    f'Morbidity in {self.year}, thousand (%)': '',
                                                    f'Mortality number in {self.yearstart} (%)': '',
                                                    f'Mortality number in {self.year} (%)': '',
                                                   
                                                   
                                                    f'DALYs in {self.yearstart}, thousand (%)': '', 
                                                    f'DALYs in {self.year}, thousand (%)': '', 
                                                    f'Annual GDP 2025–50, 2024 U.S.$ billion (global %)': '',
                                                    f'Annual population 2025–50, millions (global %)': ''}])],
                            ignore_index=True)

       
        for income_group in income_groups:
           
            income_group_df = df[df['Income Group'] == income_group]
            df_yearstart = df[df['year'] == self.yearstart]
            df_year = df[df['year'] == self.year]
            income_group_df_yearstart = income_group_df[income_group_df['year'] == self.yearstart]
            income_group_df_year = income_group_df[income_group_df['year'] == self.year]

           
            GDPloss_val_total = income_group_df_year['GDPloss_val'].sum()
            GDPloss_val_global = df_year['GDPloss_val'].sum()
            GDPloss_percentage = (GDPloss_val_total / GDPloss_val_global) * 100
            Economic_burden_total = f"{self._format_number(GDPloss_val_total / billion_factor, 0)}({self._format_number(GDPloss_percentage, decimal)}%)"

           
            morbidities_yearstart = (income_group_df_yearstart['POP'] * income_group_df_yearstart['Morbidity']).sum()
            morbidities_yearstart_global = (df_yearstart['POP'] * df_yearstart['Morbidity']).sum()
            morbidities_yearstart_percentage = (morbidities_yearstart / morbidities_yearstart_global) * 100
            morbidities_yearstart_total = f"{self._format_number(morbidities_yearstart/thousand_factor, 0)}({self._format_number(morbidities_yearstart_percentage, decimal)}%)"

            morbidities_year = (income_group_df_year['POP'] * income_group_df_year['Morbidity']).sum()
            morbidities_year_global = (df_year['POP'] * df_year['Morbidity']).sum()
            morbidities_year_percentage = (morbidities_year / morbidities_year_global) * 100
            morbidities_year_total = f"{self._format_number(morbidities_year/thousand_factor, 0)}({self._format_number(morbidities_year_percentage, decimal)}%)"

           
            mortalities_yearstart = (income_group_df_yearstart['POP'] * income_group_df_yearstart['Mortality']).sum()
            mortalities_yearstart_global = (df_yearstart['POP'] * df_yearstart['Mortality']).sum()
            mortalities_yearstart_percentage = (mortalities_yearstart / mortalities_yearstart_global) * 100
            mortalities_yearstart_total = f"{self._format_number(mortalities_yearstart, 0)}({self._format_number(mortalities_yearstart_percentage, decimal)}%)"

            mortalities_year = (income_group_df_year['POP'] * income_group_df_year['Mortality']).sum()
            mortalities_year_global = (df_year['POP'] * df_year['Mortality']).sum()
            mortalities_year_percentage = (mortalities_year / mortalities_year_global) * 100
            mortalities_year_total = f"{self._format_number(mortalities_year, 0)}({self._format_number(mortalities_year_percentage, decimal)}%)"

           
            dalys_yearstart = (income_group_df_yearstart['POP'] * income_group_df_yearstart['DALYs']).sum()
            dalys_yearstart_global = (df_yearstart['POP'] * df_yearstart['DALYs']).sum()
            dalys_yearstart_percentage = (dalys_yearstart / dalys_yearstart_global) * 100
            dalys_yearstart_total = f"{self._format_number(dalys_yearstart/thousand_factor, 0)}({self._format_number(dalys_yearstart_percentage, decimal)}%)"

            dalys_year = (income_group_df_year['POP'] * income_group_df_year['DALYs']).sum()
            dalys_year_global = (df_year['POP'] * df_year['DALYs']).sum()
            dalys_year_percentage = (dalys_year / dalys_year_global) * 100
            dalys_year_total = f"{self._format_number(dalys_year/thousand_factor, 0)}({self._format_number(dalys_year_percentage,decimal)}%)"

           
            sfGDP_val_total = income_group_df_year['GDP'].sum()
            sfGDP_val_global = df_year['GDP'].sum()
            sfGDP_percentage = (sfGDP_val_total / sfGDP_val_global) * 100
            sfGDP_total = f"{self._format_number(sfGDP_val_total / billion_factor / year_factor, 0)}({self._format_number(sfGDP_percentage, decimal)}%)"

            dfPOP_val_total = income_group_df['POP'].sum()
            dfPOP_val_global = df['POP'].sum()
            dfPOP_percentage = (dfPOP_val_total / dfPOP_val_global) * 100
            dfPOP_total = f"{self._format_number(dfPOP_val_total / million_factor / year_factor, 0)}({self._format_number(dfPOP_percentage, decimal)}%)"

           
            summary = pd.concat([summary, pd.DataFrame([{'Group': income_group,
                                                        'Economic costs, 2024 U.S.$ billion (%)': Economic_burden_total,
                                                       
                                                       
                                                        f'Morbidity in {self.yearstart}, thousand (%)': morbidities_yearstart_total,
                                                        f'Morbidity in {self.year}, thousand (%)': morbidities_year_total,
                                                        f'Mortality number in {self.yearstart} (%)': mortalities_yearstart_total,
                                                        f'Mortality number in {self.year} (%)': mortalities_year_total,
                                                       
                                                       
                                                        f'DALYs in {self.yearstart}, thousand (%)': dalys_yearstart_total, 
                                                        f'DALYs in {self.year}, thousand (%)': dalys_year_total, 
                                                        f'Annual GDP 2025–50, 2024 U.S.$ billion (global %)': sfGDP_total,
                                                        f'Annual population 2025–50, millions (global %)': dfPOP_total}])],
                                ignore_index=True)
            raw_summary = pd.concat([raw_summary,pd.DataFrame([{
                'Group': income_group,
                'Economic_cost_val': GDPloss_val_total,
                'Economic_cost_percent': GDPloss_percentage,
                f'Morbidity_{self.yearstart}_val': morbidities_yearstart,
                f'Morbidity_{self.yearstart}_percent': morbidities_yearstart_percentage,
                f'Morbidity_{self.year}_val': morbidities_year,
                f'Morbidity_{self.year}_percent': morbidities_year_percentage,
                f'Mortality_{self.yearstart}_val': mortalities_yearstart,
                f'Mortality_{self.yearstart}_percent': mortalities_yearstart_percentage,
                f'Mortality_{self.year}_val': mortalities_year,
                f'Mortality_{self.year}_percent': mortalities_year_percentage,
                f'DALYs_{self.yearstart}_val': dalys_yearstart,
                f'DALYs_{self.yearstart}_percent': dalys_yearstart_percentage,
                f'DALYs_{self.year}_val': dalys_year,
                f'DALYs_{self.year}_percent': dalys_year_percentage,
                'Annual_GDP_val': sfGDP_val_total,
                'Annual_GDP_percent': sfGDP_percentage,
                'Annual_population_val': dfPOP_val_total,
                'Annual_population_percent': dfPOP_percentage
            }])], ignore_index=True)

       
        GDPloss_val_total = df_year['GDPloss_val'].sum()
        GDPloss_val_global = df_year['GDPloss_val'].sum()
        GDPloss_percentage = (GDPloss_val_total / GDPloss_val_global) * 100
        Economic_burden_total = f"{self._format_number(GDPloss_val_total / billion_factor, 0)}({self._format_number(GDPloss_percentage,decimal)}%)"

       
        morbidities_yearstart = (df_yearstart['POP'] * df_yearstart['Morbidity']).sum()
        morbidities_yearstart_global = (df_yearstart['POP'] * df_yearstart['Morbidity']).sum()
        morbidities_yearstart_percentage = (morbidities_yearstart / morbidities_yearstart_global) * 100
        morbidities_yearstart_total = f"{self._format_number(morbidities_yearstart/thousand_factor, 0)}({self._format_number(morbidities_yearstart_percentage,decimal)}%)"

        morbidities_year = (df_year['POP'] * df_year['Morbidity']).sum()
        morbidities_year_global = (df_year['POP'] * df_year['Morbidity']).sum()
        morbidities_year_percentage = (morbidities_year / morbidities_year_global) * 100
        morbidities_year_total = f"{self._format_number(morbidities_year/thousand_factor, 0)}({self._format_number(morbidities_year_percentage, decimal)}%)"

       
        mortalities_yearstart = (df_yearstart['POP'] * df_yearstart['Mortality']).sum()
        mortalities_yearstart_global = (df_yearstart['POP'] * df_yearstart['Mortality']).sum()
        mortalities_yearstart_percentage = (mortalities_yearstart / mortalities_yearstart_global) * 100
        mortalities_yearstart_total = f"{self._format_number(mortalities_yearstart, 0)}({self._format_number(mortalities_yearstart_percentage,decimal)}%)"

        mortalities_year = (df_year['POP'] * df_year['Mortality']).sum()
        mortalities_year_global = (df_year['POP'] * df_year['Mortality']).sum()
        mortalities_year_percentage = (mortalities_year / mortalities_year_global) * 100
        mortalities_year_total = f"{self._format_number(mortalities_year, 0)}({self._format_number(mortalities_year_percentage,decimal)}%)"

       
        dalys_yearstart = (df_yearstart['POP'] * df_yearstart['DALYs']).sum()
        dalys_yearstart_global = (df_yearstart['POP'] * df_yearstart['DALYs']).sum()
        dalys_yearstart_percentage = (dalys_yearstart / dalys_yearstart_global) * 100
        dalys_yearstart_total = f"{self._format_number(dalys_yearstart/thousand_factor, 0)}({self._format_number(dalys_yearstart_percentage,decimal)}%)"

        dalys_year = (df_year['POP'] * df_year['DALYs']).sum()
        dalys_year_global = (df_year['POP'] * df_year['DALYs']).sum()
        dalys_year_percentage = (dalys_year / dalys_year_global) * 100
        dalys_year_total = f"{self._format_number(dalys_year/thousand_factor, 0)}({self._format_number(dalys_year_percentage,decimal)}%)"

       
        sfGDP_val_total = df_year['GDP'].sum()
        sfGDP_val_global = df_year['GDP'].sum()
        sfGDP_percentage = (sfGDP_val_total / sfGDP_val_global) * 100
        sfGDP_total = f"{self._format_number(sfGDP_val_total / billion_factor / year_factor, 0)}({self._format_number(sfGDP_percentage, decimal)}%)"

        dfPOP_val_total = df['POP'].sum()
        dfPOP_val_global = df['POP'].sum()
        dfPOP_percentage = (dfPOP_val_total / dfPOP_val_global) * 100
        dfPOP_total = f"{self._format_number(dfPOP_val_total / million_factor / year_factor, 0)}({self._format_number(dfPOP_percentage,decimal)}%)"

       
        total_row = pd.DataFrame({
            'Group': ['Total'],
            'Economic costs, 2024 U.S.$ billion (%)': [Economic_burden_total],
           
           
            f'Morbidity in {self.yearstart}, thousand (%)': [morbidities_yearstart_total],
            f'Morbidity in {self.year}, thousand (%)': [morbidities_year_total],
            f'Mortality number in {self.yearstart} (%)': [mortalities_yearstart_total], 
            f'Mortality number in {self.year} (%)': [mortalities_year_total], 
           
           
            f'DALYs in {self.yearstart}, thousand (%)': [dalys_yearstart_total], 
            f'DALYs in {self.year}, thousand (%)': [dalys_year_total], 
            f'Annual GDP 2025–50, 2024 U.S.$ billion (global %)': [sfGDP_total],
            f'Annual population 2025–50, millions (global %)': [dfPOP_total]
        })
        raw_summary = pd.concat([raw_summary,pd.DataFrame([{
            'Group': 'Total',
            'Economic_cost_val': GDPloss_val_total,
            'Economic_cost_percent': GDPloss_percentage,
            f'Morbidity_{self.yearstart}_val': morbidities_yearstart,
            f'Morbidity_{self.yearstart}_percent': morbidities_yearstart_percentage,
            f'Morbidity_{self.year}_val': morbidities_year,
            f'Morbidity_{self.year}_percent': morbidities_year_percentage,
            f'Mortality_{self.yearstart}_val': mortalities_yearstart,
            f'Mortality_{self.yearstart}_percent': mortalities_yearstart_percentage,
            f'Mortality_{self.year}_val': mortalities_year,
            f'Mortality_{self.year}_percent': mortalities_year_percentage,
            f'DALYs_{self.yearstart}_val': dalys_yearstart,
            f'DALYs_{self.yearstart}_percent': dalys_yearstart_percentage,
            f'DALYs_{self.year}_val': dalys_year,
            f'DALYs_{self.year}_percent': dalys_year_percentage,
            'Annual_GDP_val': sfGDP_val_total,
            'Annual_GDP_percent': sfGDP_percentage,
            'Annual_population_val': dfPOP_val_total,
            'Annual_population_percent': dfPOP_percentage
        }])], ignore_index=True)
        summary = pd.concat([summary, total_row], ignore_index=True)

       
        final_order = ['By World Bank region', 'East Asia and Pacific', 'Europe and Central Asia',
                    'Latin America and Caribbean', 'Middle East and North Africa', 'North America',
                    'South Asia', 'Sub-Saharan Africa', 'By World Bank income group', 'Low income',
                    'Lower middle income', 'Upper middle income', 'High income', 'Total']

   
        final_summary = pd.DataFrame(columns=summary.columns) 
        for group in final_order:
            row = summary[summary['Group'] == group]
            if not row.empty:
                final_summary = pd.concat([final_summary, row], ignore_index=True)
            else:
           
                final_summary = pd.concat([final_summary, pd.DataFrame([{'Group': group,
                                                                'Economic costs, 2024 U.S.$ billion (%)': '',
                                                               
                                                               
                                                                f'Morbidity in {self.yearstart}, thousand (%)': '',
                                                                f'Morbidity in {self.year}, thousand (%)': '',
                                                                f'Mortality number in {self.yearstart} (%)':'',  #Added
                                                                f'Mortality number in {self.year} (%)':'',   #Added
                                                               
                                                               
                                                                f'DALYs in {self.yearstart}, thousand (%)': '', 
                                                                f'DALYs in {self.year}, thousand (%)': '', 
                                                                f'Annual GDP 2025–50, 2024 U.S.$ billion (global %)': '',
                                                                f'Annual population 2025–50, millions (global %)': ''}])], ignore_index=True)
       
        if mode=="exp1_baseline":
            raw_filename = "table9_raw.csv"
        elif mode=="exp2_varparams":
            raw_filename = "table10_raw.csv"
        elif mode=="exp3_vardiscount2":
            raw_filename = "table11_raw.csv"
        elif mode=="exp4_vardiscount0":
            raw_filename = "table12_raw.csv"
        
        raw_output_path = os.path.join(self.output_dir, raw_filename)
        raw_summary.to_csv(raw_output_path, index=False)
        print(f"原始汇总数据已保存到: {raw_output_path}")
        if mode=="exp1_baseline":
            self.logger.info("table9:Macroeconomic and lifetime disease costs measured in Mortality and Morbidity by region and country SDI value quantile groupings")
           
            self.save_grouped_data(final_summary, filename="table9.csv")
        elif mode=="exp2_varparams":
            self.logger.info("table10:Macroeconomic and lifetime disease costs measured in Mortality and Morbidity by region and country SDI value quantile groupings with different parameters")
           
            self.save_grouped_data(final_summary, filename="table 10.csv")
        elif mode=="exp3_vardiscount2":
            self.logger.info("table11:Macroeconomic and lifetime disease costs measured in Mortality and Morbidity by region and country SDI value quantile groupings with a discount rate of 2%")
           
            self.save_grouped_data(final_summary, filename="table 11.csv")
        elif mode=="exp4_vardiscount0":
            self.logger.info("table12:Macroeconomic and lifetime disease costs measured in Mortality and Morbidity by region and country SDI value quantile groupings with a discount rate of 0%")
           
            self.save_grouped_data(final_summary, filename="table 12.csv")
            
    def run_full_process(self):
        self._table_discount_visual(mode="exp1_baseline")
        self._table_discount_visual(mode="exp2_varparams")
        self._table_discount_visual(mode="exp3_vardiscount2")
        self._table_discount_visual(mode="exp4_vardiscount0")

        self._table_discount_visual_full_format(mode="exp1_baseline")
        self._table_discount_visual_full_format(mode="exp2_varparams")
        self._table_discount_visual_full_format(mode="exp3_vardiscount2")
        self._table_discount_visual_full_format(mode="exp4_vardiscount0")

        self._table_eco_agg_visual(mode="exp1_baseline")
        self._table_eco_agg_visual(mode="exp2_varparams")
        self._table_eco_agg_visual(mode="exp3_vardiscount2")
        self._table_eco_agg_visual(mode="exp4_vardiscount0")

        self._table_mor_agg_visual(mode="exp1_baseline")
        self._table_mor_agg_visual(mode="exp2_varparams")
        self._table_mor_agg_visual(mode="exp3_vardiscount2")
        self._table_mor_agg_visual(mode="exp4_vardiscount0")
        print("数据处理完成")

