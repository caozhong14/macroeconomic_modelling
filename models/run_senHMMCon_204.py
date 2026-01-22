import os
import sys
import logging
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from itertools import accumulate


sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))


from models.HMMDataLoader import HMMDataLoader
from models.HMM import HMM
from models.senHMM import senHMM
from models.senHMMController import senHMMController

class SensitivityAnalyzer:
    def __init__(self, json_dir, output_folder, n_runs,config):
        self.json_dir = json_dir
        self.output_folder = output_folder
        self.n_runs = n_runs
        self.config = config
        self.logger = self._setup_logger()
        self.country_name = None 
        self.country_data = {} 
       

    def _setup_logger(self):
        """
        设置日志记录器。

        Returns
        -------
        logging.Logger
            配置好的日志记录器
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

       
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

       
        output_log_dir = os.path.join(os.path.dirname(__file__), 'output')
        if not os.path.exists(output_log_dir):
            os.makedirs(output_log_dir)
        log_file_path = os.path.join(output_log_dir, 'process.log')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def process_and_merge_json_results(self):
        """
        遍历 json_dir 中的所有 JSON 文件，对每个文件进行处理，并合并结果。
        """
        self.logger.info(f"开始处理目录 {self.json_dir} 中的 JSON 文件，并进行预测。")

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

       
        merged_results = {
            "exp1_baseline": [],
            "exp2_varparams": [],
            "exp3_vardiscount2": [],
            "exp4_vardiscount0": []
        }

       
        json_files = [f for f in os.listdir(self.json_dir) if f.endswith('.json')]
        if not json_files:
            self.logger.error(f"目录 {self.json_dir} 中没有找到 JSON 文件。")
            return

        for json_file in tqdm(json_files, desc="Processing JSON files"):
            file_path = os.path.join(self.json_dir, json_file)
            try:
               
                self.logger.info(f"加载数据: {file_path}")
                loader = HMMDataLoader(file_path)
                df_data = loader.df
            except Exception as e:
                self.logger.error(f"加载文件 {file_path} 时出错: {e}")
                continue

           
            self.country_name = os.path.splitext(json_file)[0] 
            self.logger.info(f"正在处理国家: {self.country_name}")

            try:
               
               
                
               
                controller = senHMMController(df_sq=df_data, config =self.config)
                result_dict = controller.run_all_scenarios(n_runs=self.n_runs)
            except Exception as e:
                self.logger.error(f"{self.country_name} 预测过程中出错: {e}")
                continue

           
            for scenario, df_stat in result_dict.items():
               
                df_stat.insert(0, 'country', self.country_name)
                df_stat.insert(1, 'scenario', scenario)
                
               
               
                
               
                merged_results[scenario].append(df_stat)

       
        all_scenarios_df = pd.DataFrame()
        for scenario, df_list in merged_results.items():
            if df_list:
                merged_df = pd.concat(df_list, ignore_index=True)
                all_scenarios_df = pd.concat([all_scenarios_df, merged_df], ignore_index=True)
                self.logger.info(f"{scenario} 结果已合并。")
            else:
                self.logger.info(f"{scenario} 没有任何数据被处理。")

       
        if not all_scenarios_df.empty:
           
            new_columns = [col.replace("q25", "lower").replace("q500", "val").replace("q975", "upper") 
                          for col in all_scenarios_df.columns]
            all_scenarios_df.columns = new_columns
            
           
            all_scenarios_df.rename(columns={
                "country": "country_codes",
                "scenario": "task"
            }, inplace=True)

       
        if not all_scenarios_df.empty:
            output_filepath = os.path.join(self.output_folder, "all_results.csv")
            try:
                all_scenarios_df.to_csv(output_filepath, index=False)
                self.logger.info(f"所有场景结果已合并并保存至: {output_filepath}")
            except Exception as e:
                self.logger.error(f"保存 {output_filepath} 时出错: {e}")
        else:
            self.logger.info("没有任何数据被处理。")
