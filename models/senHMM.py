import numpy as np
import pandas as pd
from HMMDataLoader import HMMDataLoader
from HMM import HMM 
import warnings
import traceback
class senHMM:
    def __init__(self, df_sq,config):
        self.df_sq = df_sq.copy()
        self.config = config.copy()
       
        self.base_model = HMM(self.df_sq,config=self.config)
        self.original_params = {
            'cd': self.base_model.params['algorithm_layer']['cd'],
            'eta1': self.base_model.params['algorithm_layer']['eta1'],
            'eta2': self.base_model.params['algorithm_layer']['eta2'],
            'eta3': self.base_model.params['algorithm_layer']['eta3'],
            'ds': self.base_model.params['algorithm_layer']['ds']
        }

    def _perturb_health_data(self, df: pd.DataFrame) -> pd.DataFrame:
    
        df_perturbed = df.copy()
       
        z = np.random.normal(0, 1)
       
        ln_deaths = np.log(df_perturbed['Deaths_val'] + 1e-16)
       
        sigma_deaths = (np.log(df_perturbed['Deaths_upper'] + 1e-16) 
                        - np.log(df_perturbed['Deaths_lower'] + 1e-16)) / 3.92
       
       
        sigma_deaths=sigma_deaths.clip(0.012,0.587) 
        ln_deaths_perturbed = ln_deaths + sigma_deaths * z
       
        df_perturbed['Deaths_val'] = np.exp(ln_deaths_perturbed)

       
        ln_morbidity = np.log(df_perturbed['Morbidity_val'] + 1e-16)
       
        sigma_morbidity = (np.log(df_perturbed['Morbidity_upper'] + 1e-16) 
                           - np.log(df_perturbed['Morbidity_lower'] + 1e-16)) / 3.92
            
        sigma_morbidity=sigma_morbidity.clip(0.012,0.587) 
        ln_morbidity_perturbed = ln_morbidity + sigma_morbidity * z
        df_perturbed['Morbidity_val'] = np.exp(ln_morbidity_perturbed)
        return df_perturbed

    def _perturb_TC(self, df: pd.DataFrame, strength: float = 100) -> pd.DataFrame:
        """
        使用 Beta 分布对 TreatmentFraction 列进行扰动：
        - 仅对第一行计算扰动参数 alpha 和 beta。
        - 从 Beta(alpha, beta) 中采样一个值，并应用于所有行。
        - strength 用于控制扰动强度，默认为 100。
        """
        df_perturbed = df.copy()
       
        first_rate = np.clip(df_perturbed.loc[0, 'TreatmentFraction'], 1e-16, 1 - 1e-16)
       
        alpha = first_rate * strength
        beta_param = (1 - first_rate) * strength
       
        sampled_value = np.random.beta(alpha, beta_param)
       
        df_perturbed['TreatmentFraction'] = np.clip(sampled_value, 0, 1)
        return df_perturbed

    def _perturb_all_parameters(self, df: pd.DataFrame):
       
        df_perturbed = self._perturb_health_data(df)
        if self.config['algorithm_layer']['exp2mode_tc_perturb']:
           
            df_perturbed=self._perturb_TC(df_perturbed)
           
        else:
            print("未对TC数据扰动！！！！")
       
        params_perturbed = {}
        for pname, pval in self.original_params.items():
            if isinstance(pval, (int, float)):
                factor = np.random.uniform(0.5, 1.5)
                params_perturbed[pname] = pval * factor
            else:
                params_perturbed[pname] = pval 

        return df_perturbed, params_perturbed

    def _run_health_sample(self, seed: int, ds=None) -> pd.DataFrame:
        """
        单次 perturb：只扰动健康数据和治疗费用数据。可选传入 ds 替换原贴现率。
        """
        np.random.seed(seed)
        df_perturbed = self._perturb_health_data(self.df_sq)

        if self.config['algorithm_layer']['exp2mode_tc_perturb']:
           
            df_perturbed=self._perturb_TC(df_perturbed)
           
        else:
            print("其余实验未对TC数据扰动！！！！")
        if ds is not None:
            params = {'ds': ds}
        else:
            params = None
       
        model = HMM(df_perturbed, params=params,config=self.config)
       
        result = model.df_result
        result['sample'] = seed
        result['mode'] = 'health'
        return result

    def _run_params_sample(self, seed: int) -> pd.DataFrame:
        """
        单次 perturb：同时扰动健康数据和其它参数。
        """
        np.random.seed(seed)
        df_perturbed, params_perturbed = self._perturb_all_parameters(self.df_sq)

        model = HMM(df_perturbed, params=params_perturbed,config=self.config)
        result = model.df_result
        result['sample'] = seed
        result['mode'] = 'params'
        return result

    def _run_direct(self, seed: int, ds=None) -> pd.DataFrame:
        """
        不扰动：使用原始数据和原始参数。
        """
             
        if ds is None:
            result = self.base_model.df_result
        else:
            self.base_model.apply_discount(discount=ds)
            result = self.base_model.df_result
        result['sample'] = seed
        result['mode'] = 'direct'
        return result
  