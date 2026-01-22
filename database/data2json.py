import os
import pandas as pd
from pandas import IndexSlice as idx
import numpy as np
import json
from tqdm import tqdm
import random


class DataProcessor:
    def __init__(self,country_code,disease,configname,startyear,
                 projectStartYear,endyear,config):
       
        self.random_seed = 42
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.country_code = country_code
        self.startyear=startyear
        self.projectStartYear =projectStartYear
        self.endyear =endyear
        self.disease = disease
        self.configname=configname
       
        self.perturb_factor=1e-16
        self.CapitalShare=config['data_layer']['CapitalShare']
        self.data_dir = f"./database/bigdata/{self.configname}"
        self.default_codebook = config['data_layer']['default_codebook']
        self.weight = config['data_layer']['weight']
        self.DW = config['data_layer']['morbid_DW_cal']
        self.alph = config['data_layer']['alph']
        self.tc_fraction = config['data_layer']['tc_fraction']
        self.physical_ppp = config['data_layer']['physical_ppp']
        self.gdp = config['data_layer']['gdp']
        self.hepc = config['data_layer']['hepc']
        self.savings = config['data_layer']['savings']
        self.population = config['data_layer']['population']
        self.labor = config['data_layer']['labor']
        self.education = config['data_layer']['education']
        self.has_required_data = config['data_layer']['has_required_data']
        missing_data_summary_columns =config['data_layer']['missing_data_summary_columns']
        self.missing_data_summary = pd.DataFrame(columns=missing_data_summary_columns)
       

    def _validate_paths(self):
       
        """验证所有必需文件路径是否存在"""
       
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"找不到数据目录: {self.data_dir}")
        if not os.path.exists(self.default_codebook):
            raise FileNotFoundError(f"找不到codebook文件: {self.default_codebook}")
        else:
            print('目录和codebook正确')
            
       
        self.data_files = {
            'alpha': self.alph[0],
            'TC fraction': self.tc_fraction[0],
            'physical PPP': self.physical_ppp[0],
            'GDP': self.gdp[0],
            'HEPC': self.hepc[0],
            'savings': self.savings[0],
            'population': self.population[0],
            'labor': self.labor[0],
            'education': self.education[0]
        }
        for name, path in self.data_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"找不到{name}数据文件: {path}")
        else:
            print('其他数据文件正确')

    def load_and_concatenate(self):
        """加载并合并指定目录下的所有CSV文件"""
        self._validate_paths()
       
        target_dir = self.data_dir
        if not os.path.isdir(target_dir):
            raise FileNotFoundError(f"Directory '{self.data_dir}' not found")

        csv_files = [f for f in os.listdir(target_dir) if f.endswith(".csv")]
        if not csv_files:
            raise ValueError(f"No CSV files found in '{self.data_dir}'")

        dataframes = []
        for filename in csv_files:
            file_path = os.path.join(target_dir, filename)
            try:
                df = pd.read_csv(file_path).reset_index(drop=True)
                dataframes.append(df)
            except pd.errors.EmptyDataError:
                print(f"跳过空文件: {filename}")
            except Exception as e:
                print(f"读取文件{filename}失败: {e}")
                raise

        self.raw_df = pd.concat(dataframes, ignore_index=True)
        return self.raw_df

    def process_with_codebook(self):
        """使用codebook进行数据处理"""
        if self.raw_df is None:
            raise ValueError("请先加载数据")
        else:
            print('concat后的healthdata数据维度是',self.raw_df.shape)

       
        cols_to_drop = ['scenario', 'rei', 'metric']
        self.processed_df = self.raw_df.drop(
            columns=[c for c in cols_to_drop if c in self.raw_df.columns]
        )

       
        try:
            codebook = pd.read_excel(self.default_codebook)
        except FileNotFoundError:
            raise FileNotFoundError(f"Codebook文件不存在于 {self.codebook_path}")
        except Exception as e:
            raise RuntimeError(f"读取codebook失败: {e}")

       
        for col in self.processed_df.columns:
            id_col = f"{col}ID"
            name_col = f"{col}Name"
            
            if id_col in codebook.columns and name_col in codebook.columns:
                codebook_sub = codebook.drop_duplicates(subset=[id_col], keep='last')
                mapping = dict(zip(codebook_sub[id_col], codebook_sub[name_col]))
                self.processed_df[col] = self.processed_df[col].map(mapping).fillna(self.processed_df[col])
                self.processed_df = self.processed_df[~self.processed_df.isin(['Nooo']).any(axis=1)]
    
            else:
                print(f"跳过缺失映射的列: {col}")
        print('codebook映射并删除Nooo后的heathdata数据维度',self.processed_df.shape)

    def create_pivot_table(self):
        """创建透视表"""
        if self.processed_df is None:
            raise ValueError("请先处理数据")

        self.pivot_df = pd.pivot_table(
            self.processed_df,
            index=['location', 'cause', 'sex', 'age', 'measure'],
            columns='year',
            values=['mean', 'upper', 'lower']
        )
        return self.pivot_df

   
   
   
   

   
   
   
   

   
   
   
   
        
   
    def filter_data(self, measures=None):
        """筛选指定数据"""
        if self.pivot_df is None:
            raise ValueError("请先创建透视表")

        available_measures = self.pivot_df.index.get_level_values('measure').unique().tolist()
        
        if measures is None:
           
            desired_measures = ['Deaths', "YLDs (Years Lived with Disability)", "YLLs (Years of Life Lost)","DALYs (Disability-Adjusted Life Years)"] if not self.DW else ['Deaths','Prevalence',"DALYs (Disability-Adjusted Life Years)"]
            
           
            measures = [measure for measure in desired_measures if measure in available_measures]
            
           
            if not measures:
                measures = available_measures
                print(f"警告: 期望的指标都不存在，使用所有可用指标: {measures}")
            else:
                print(f"使用存在的指标: {measures}")

        self.pivot_df = self.pivot_df.loc[
            idx[self.country_code, self.disease, :, :, measures],
            idx[['mean', 'upper', 'lower'], :]
        ]
        
        return self.pivot_df

    def _extract_values(self, unstacked, idx_group, year, measure):
        """提取指定指标的mean、upper和lower值"""
        return {
            'mean': unstacked.loc[idx_group, ('mean', year, measure)],
            'upper': unstacked.loc[idx_group, ('upper', year, measure)],
            'lower': unstacked.loc[idx_group, ('lower', year, measure)]
        }

    def morbidity(self):
        unstacked = self.pivot_df.unstack('measure')
        years = unstacked.columns.get_level_values(1).unique()
        self.morbidity_data = []
        available_measures = self.pivot_df.index.get_level_values('measure').unique().tolist()

        if 'Deaths' in available_measures and not self.DW:  
            for idx_group in unstacked.index:
                location, cause, sex, age = idx_group
                for year in years:
                    try:
                        deaths_values = self._extract_values(unstacked, idx_group, year, 'Deaths')
                        ylds_values = self._extract_values(unstacked, idx_group, year, 'YLDs (Years Lived with Disability)')
                        ylls_values = self._extract_values(unstacked, idx_group, year, 'YLLs (Years of Life Lost)')

                       
                       
                       
                        mu_death = np.log(deaths_values['mean'] + self.perturb_factor)
                        sigma_death = (np.log(deaths_values['upper'] + self.perturb_factor) - np.log(deaths_values['lower'] + self.perturb_factor)) / 3.92
                        mu_ylds = np.log(ylds_values['mean'] + self.perturb_factor)
                        sigma_ylds = (np.log(ylds_values['upper'] + self.perturb_factor) - np.log(ylds_values['lower'] + self.perturb_factor)) / 3.92
                        mu_ylls = np.log(ylls_values['mean'] + self.perturb_factor)
                        sigma_ylls = (np.log(ylls_values['upper'] + self.perturb_factor) - np.log(ylls_values['lower'] + self.perturb_factor)) / 3.92
                        
                       
                       
                       
                        deaths_samples = np.random.lognormal(mu_death, sigma_death, 1000)
                        ylds_samples = np.random.lognormal(mu_ylds, sigma_ylds, 1000)
                        ylls_samples = np.random.lognormal(mu_ylls, sigma_ylls, 1000)
                        
                       
                       
                        morbidity = (ylds_samples * deaths_samples) / ylls_samples
                        
                       
                        lower = np.quantile(morbidity, 0.025)
                        val = np.quantile(morbidity, 0.5)
                        upper = np.quantile(morbidity, 0.975)
                        
                        self.morbidity_data.append({
                            'location': location,
                            'cause': cause,
                            'sex': sex,
                            'age': age,
                            'measure': 'Morbidity',
                            'year': year,
                            'mean': val,
                            'upper': upper,
                            'lower': lower
                        })
                    except KeyError:
                        continue 
                    except Exception as e:
                        print(f"Error processing {idx_group}, {year}: {e}")
                        continue
        else:
           
            try:
                weights_df = pd.read_excel(self.weight)
                weights = weights_df[weights_df['disease'] == self.disease]['weight'].values[0]
                DW= weights_df[weights_df['disease'] == self.disease]['DW'].values[0]
            except Exception as e:
                raise ValueError(f"权重数据加载失败: {e}")
            for idx_group in unstacked.index:
                location, cause, sex, age = idx_group
                for year in years:
                    try:
                       
                        preva_mean = unstacked.loc[idx_group, ('mean', year, 'Prevalence')]
                        preva_upper = unstacked.loc[idx_group, ('upper', year, 'Prevalence')]
                        preva_lower = unstacked.loc[idx_group, ('lower', year, 'Prevalence')]
                    
                       
                       
                       
                       
                        mu_prevalence = np.log(preva_mean + self.perturb_factor)         
                        sigma_prevalence = (np.log(preva_upper + self.perturb_factor) -np.log(preva_lower + self.perturb_factor)) / 3.92
                        prevalence_sample = np.random.lognormal(mu_prevalence, sigma_prevalence, 1000)
                       
                        morbidity = prevalence_sample * weights

                       
                       
                       
                       
                    
                       
                       
                       
                       
                       
                       
                       
                       
                       

                       
                        lower = np.quantile(morbidity, 0.025)
                        val = np.quantile(morbidity, 0.5)
                        upper = np.quantile(morbidity, 0.975)

                        self.morbidity_data.append({
                        'location': location,
                        'cause': cause,
                        'sex': sex,
                        'age': age,
                        'measure': 'Morbidity',
                        'year': year,
                        'mean': val,
                        'upper': upper,
                        'lower': lower
                    })
                    except KeyError as e:
                        print(f"缺失Prevalence数据: {e}")
                    except Exception as e:
                        print(f"Error processing {idx_group}, {year}: {e}")

        return  self.morbidity_data

    def concat_morbid(self):
        """拼接morbidity数据"""
        self.morbidity_df = pd.DataFrame(self.morbidity_data)
        self.morbidity_pivot = self.morbidity_df.pivot_table(
        index=['location', 'cause', 'sex', 'age', 'measure'],
        columns='year',
        values=['mean', 'upper', 'lower']
)

        self.concatenated_df = pd.concat([self.pivot_df, self.morbidity_pivot]).sort_index()
        return self.concatenated_df
    
    def _check_measure_exists(self, measure_name):
        return measure_name in self.concatenated_df.index.get_level_values('measure')

    def datatojson(self):
        """json数据然后从各个文件中提取目标数值"""
       
       
       

        country_code = self.country_code
        self.json_data = {}
        data = pd.read_csv(self.alph)
        if all(not item for item in data['Country Code'] == self.country_code):
            self.json_data['CapitalShare'] =self.CapitalShare
           
        else:
            self.json_data['CapitalShare'] = data[data['Country Code'] == self.country_code]['alpha'].values[0]

       
        data = pd.read_csv(self.tc_fraction)
        self.json_data['TreatmentFraction'] = data[data['Country Code'] == self.country_code][self.disease].values[0]

       
        data = pd.read_csv(self.physical_ppp)
        if all(not item for item in data['Country Code'] == self.country_code):
           
           
            print(self.country_code+"没有初始资本数据！！！！")
            self.has_required_data = False
            self.log_missing_data('InitialCapitalStock')
        else:
           
            self.json_data['InitialCapitalStock'] = data[data['Country Code'] == self.country_code][str(self.startyear)].values[0]*1e6

        for year in tqdm(range(self.projectStartYear, self.endyear), desc="Processing Years"):
            year_str = str(year)
            year = int(year)
           
            data =  pd.read_csv(self.gdp)
            self.json_data[year_str] = {}
            if all(not item for item in data['Country Code'] == self.country_code):
               
                print(self.country_code+"没有GDP数据！！！！")
                self.has_required_data = False
                self.log_missing_data('GDP')
            else:
                self.json_data[year_str]['GDP'] = data[data['Country Code'] == country_code][year_str].values[0]

           
            data = pd.read_csv(self.hepc)
            if all(not item for item in data['Country Code'] == self.country_code):
               
                print(self.country_code+"没有健康支出数据！！！！")
                self.has_required_data = False
                self.log_missing_data('HealthExpenditure')
            else:
                self.json_data[year_str]['HealthExpenditure'] = data[data['Country Code'] == country_code][year_str].values[0]

           
            data = pd.read_csv(self.savings)
            if all(not item for item in data['Country Code'] == self.country_code):
               
                print(self.country_code+"没有储蓄率数据！！！！")
                self.has_required_data = False
                self.log_missing_data('SavingRate')
            else:
                self.json_data[year_str]['SavingRate'] = data[data['Country Code'] == country_code][year_str].values[0]

           
            self.json_data[year_str]['F'] = {}
            self.json_data[year_str]['M'] = {}

            for sex in ['F', 'M']:
               
               
                for age_group in ['d00', 'd05', 'd10', 'd15', 'd20', 'd25', 'd30', 'd35', 'd40', 'd45', 'd50', 'd55', 'd60', 'd65',
                                   'd70', 'd75', 'd80', 'd85', 'd90', 'd95']:
                    self.json_data[year_str][sex][age_group] = {}

                   
                    data = pd.read_csv(self.population)
                    age_data = data.loc[(data['Country Code'] == country_code) & (data['sex'] == sex) & (data['age'] == age_group)]
                    self.json_data[year_str][sex][age_group]['Population'] = age_data[year_str].values[0].astype(int) if not age_data.empty else None

                   
                    data = pd.read_csv(self.labor)
                    age_data = data.loc[(data['Country Code'] == country_code) & (data['sex'] == sex) & (data['age'] == age_group)]
                    self.json_data[year_str][sex][age_group]['LaborRate'] = age_data[year_str].values[0] if not age_data.empty else None

                   
                    data = pd.read_csv(self.education)
                    age_data = data.loc[(data['Country Code'] == country_code) & (data['sex'] == sex) & (data['age'] == age_group)]
                    self.json_data[year_str][sex][age_group]['EducationYear'] = age_data[year_str].values[0] if not age_data.empty else None

                   
                    data = self.concatenated_df.copy()
                   
                    age_data = data.loc[idx[self.country_code,self.disease,sex,age_group,'DALYs (Disability-Adjusted Life Years)'],
                                :
                            ]
                    self.json_data[year_str][sex][age_group]['DALYs'] = age_data.loc[
                                idx['mean'],idx[year]]
                    
                   
                    self.json_data[year_str][sex][age_group]['Deaths'] = {}
                    if self._check_measure_exists('Deaths'):
                        try:
                            data = self.concatenated_df.copy()
                            age_data = data.loc[
                                idx[self.country_code,self.disease,sex,age_group,'Deaths'],
                                :
                            ]
                   
                   
                   
                   
                            self.json_data[year_str][sex][age_group]['Deaths']['val'] = age_data.loc[
                                idx['mean'],
                                idx[year]]
                            
                            self.json_data[year_str][sex][age_group]['Deaths']['upper'] = age_data.loc[
                                idx['upper'],
                                idx[year]]
                            
                            self.json_data[year_str][sex][age_group]['Deaths']['lower'] = age_data.loc[
                                idx['lower'],
                                idx[year]]
                        except KeyError as e:
                            print(f"部分Death数据缺失: {e}")
                    
                    else:
                        print("没有Deaths指标，设置为0")
                        self.json_data[year_str][sex][age_group]['Deaths'].update({
                            'val': 0,
                            'upper': 0,
                            'lower': 0
                        })
                   
                    self.json_data[year_str][sex][age_group]['Morbidity'] = {}
                    data = self.concatenated_df.copy()
                    age_data = data.loc[
                        idx[self.country_code,self.disease,sex,age_group,'Morbidity'],
                        :
                    ]
                    self.json_data[year_str][sex][age_group]['Morbidity']['val'] = age_data.loc[
                        idx['mean'],
                        idx[year]]
                    
                    self.json_data[year_str][sex][age_group]['Morbidity']['upper'] = age_data.loc[
                        idx['upper'],
                        idx[year]]
                    
                    self.json_data[year_str][sex][age_group]['Morbidity']['lower'] = age_data.loc[
                        idx['lower'],
                        idx[year]]

        return self.json_data
    
    @staticmethod
    def convert_to_python_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def save_to_json(self):
        if not self.has_required_data:
            print(f"Skipping JSON saving for {self.country_code} due to missing data.")
            return

       
        output_dir = f"./database/json_output/{self.configname}"
        self.filename = f"{output_dir}/{self.country_code}.json"

       
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")
            except OSError as e:
                print(f"Error creating directory {output_dir}: {e}")
                return 
       
        self.save_data = json.loads(json.dumps(self.json_data, default=self.convert_to_python_types))

       
        with open(self.filename, 'w') as json_file:
            json.dump(self.save_data, json_file, indent=4)

        print(f"JSON data has been saved to {self.filename}")

    def log_missing_data(self, data_type):
        with open('./database/missing_data_log.txt', 'a') as log_file:
            log_file.write(f"{self.country_code} 缺失数据类型: {data_type}\n")
        
       
        if self.country_code not in self.missing_data_summary['country_code'].values:
            new_row = {'country_code': self.country_code, 'InitialCapitalStock': 0, 'GDP': 0, 'HealthExpenditure': 0, 'SavingRate': 0, 'Deaths': 0, 'Morbidity': 0}
            
           
           
            new_row_df = pd.DataFrame([new_row]) 
            self.missing_data_summary = pd.concat([self.missing_data_summary, new_row_df], ignore_index=True)
            

        self.missing_data_summary.loc[self.missing_data_summary['country_code'] == self.country_code, data_type] = 1

    def run_full_process(self):
        """统一接口，执行完整的数据处理流程"""
        self.load_and_concatenate()
        self.process_with_codebook()
        self.create_pivot_table()
        self.filter_data()
        self.morbidity()
        self.concat_morbid()
        self.datatojson()
        if self.has_required_data:
            self.save_to_json()
        else:
            print(f"Data processing completed for {self.country_code}, but JSON saving was skipped due to missing data.")
        
       
        summary_row = self.missing_data_summary.sum(numeric_only=True)
        summary_row['country_code'] = '总计'
        self.missing_data_summary = pd.concat([self.missing_data_summary, summary_row], ignore_index=True)
       

       
        print(self.missing_data_summary)




    
    
