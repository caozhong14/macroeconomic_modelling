import os
import logging
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from senHMM import senHMM


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class senHMMController:
    """
    用于控制运行 senHMM 的类。
    负责：
      1. 多次运行 (如 1000次) 并行或串行
      2. 不同 ds 配置、不同模式（health / params）
      3. 统计分位数结果等
    """

    def __init__(self, df_sq, config):
        """
        Parameters
        ----------
        df_sq : pd.DataFrame
            原始数据 DataFrame
        n_jobs : int
            并行进程/线程数
        """
        self.sen = senHMM(df_sq,config) 
        self.n_jobs = config['algorithm_layer']['n_jobs'] 
        self.config = config

    def _run_multiple(
        self, 
        mode='health', 
        ds=None, 
        n_runs=2
    ) -> pd.DataFrame:
        """
        外部函数：多次调用 senHMM 的单次运行方法，并行执行。

        Parameters
        ----------
        mode : str
            'health' / 'params' / 'direct'
        ds : float or None
            当 mode='health' 时，可指定新的贴现率
        n_runs : int
            扰动次数
        
        Returns
        -------
        df_all : pd.DataFrame
            合并后的所有单次运行结果
        """
        if mode not in ('health', 'params', 'direct'):
            raise ValueError(f"mode='{mode}'不支持，只能是health, params 或 direct")

        logger.info(f"Start _run_multiple with mode={mode}, ds={ds}, n_runs={n_runs}")
        t0 = time.time()

       
        def run_one(seed):
            logger.debug(f"    run_one(seed={seed}) start.")
            if mode == 'health':
                result = self.sen._run_health_sample(seed, ds=ds)
            elif mode == 'params':
                result = self.sen._run_params_sample(seed)
            else:
                result = self.sen._run_direct(seed, ds=ds)
            logger.debug(f"    run_one(seed={seed}) done.")
            return result

       
        with tqdm_joblib(tqdm(desc=f"Running {mode} runs", total=n_runs)) as progress_bar:
            results_list = Parallel(n_jobs=self.n_jobs)(
                delayed(run_one)(i) for i in range(n_runs)
            )

       
        df_all = pd.concat(results_list, ignore_index=True)
        t1 = time.time()
        logger.info(f"End _run_multiple with mode={mode}. Time used: {t1 - t0:.2f} sec.")
        return df_all

    def _get_quantile_stats(
        self, 
        df: pd.DataFrame, 
        metrics=None, 
        quantiles=[0.025, 0.5, 0.975],
        group_cols=['year']
    ) -> pd.DataFrame:
        """
        对指定指标在 df 上进行分位数统计。
        默认按 year 分组，如果还想按 mode 区分，可以 group_cols=['year','mode']。

        Parameters
        ----------
        df : pd.DataFrame
            多次运行合并的数据表
        metrics : list of str
            要统计的列，比如 ['GDPShare', 'GDPloss', 'GDPper','Lcontri','Kcontri'] 等
        quantiles : list
            要计算的分位数
        group_cols : list
            分组列，默认只按 'year'；可改为 ['year','mode'].

        Returns
        -------
        df_quantiles : pd.DataFrame
            包含分位数统计结果的表
        """
        if metrics is None:
            metrics = ['GDPShare', 'GDPloss', 'GDPper', 'Lcontri', 'Kcontri'] 

        logger.debug(f"Calculating quantiles for metrics={metrics} grouped by={group_cols}.")
        df_stats = df.groupby(group_cols)[metrics].quantile(quantiles)
       
        df_stats = df_stats.unstack(level=-1)

       
        new_cols = []
        for col_tuple in df_stats.columns:
            metric_name = col_tuple[0]  
            qval = col_tuple[1]        
            new_cols.append(f"{metric_name}_q{int(qval*1000)}")

        df_stats.columns = new_cols
        df_stats = df_stats.reset_index()
        return df_stats

    def run_all_scenarios(self, n_runs=100):
        """
        运行所有你指定的场景：
          1) baseline数据: 干扰健康数据 n_runs 次 ds=0.03
          2) 敏感分析数据: 干扰所有参数 n_runs 次
          3) ds 高数据: 干扰健康数据 n_runs 次 ds=0.02
          4) ds 低数据: 干扰健康数据 n_runs 次 ds=0.0
        
        并对各自结果计算分位数统计。
        
        Returns
        -------
        dict
            包含4个结果DataFrame的字典，如:
            {
              'exp1_baseline': df_baseline_stats,
              'exp2_varparams': df_sens_stats,
              'exp3_vardiscount2': df_high_stats,
              'exp4_vardiscount0': df_low_stats
            }
        """
        logger.info("=== Start run_all_scenarios ===")
        overall_start_time = time.time()

       
        logger.info("1) baseline数据 (health), ds=0.03")
        t0 = time.time()
       
       
        df_base = self._run_multiple(mode=self.config['algorithm_layer']['exp1mode'], ds=0.03, n_runs=n_runs)
        df_base_stats = self._get_quantile_stats(df_base)
        t1 = time.time()
        logger.info(f"   baseline 数据完成，耗时: {t1 - t0:.2f} 秒")

       
        logger.info("2) 敏感分析数据 (params)")
        t0 = time.time()
        df_sens = self._run_multiple(mode=self.config['algorithm_layer']['exp2mode'], ds=None, n_runs=n_runs)
        df_sens_stats = self._get_quantile_stats(df_sens)
        t1 = time.time()
        logger.info(f"   敏感分析 数据完成，耗时: {t1 - t0:.2f} 秒")

       
        logger.info("3) 折旧率 (direct), ds=0.02")
        t0 = time.time()
        df_high = self._run_multiple(mode=self.config['algorithm_layer']['exp3mode'], ds=0.02, n_runs=n_runs)
        df_high_stats = self._get_quantile_stats(df_high)
        t1 = time.time()
        logger.info(f"   折旧率 数据完成，耗时: {t1 - t0:.2f} 秒")

       
        logger.info("4) 折旧率 (direct), ds=0.0")
        t0 = time.time()
        df_low = self._run_multiple(mode=self.config['algorithm_layer']['exp4mode'], ds=0.0, n_runs=n_runs)
        df_low_stats = self._get_quantile_stats(df_low)
        t1 = time.time()
        logger.info(f"   折旧率 数据完成，耗时: {t1 - t0:.2f} 秒")

        overall_end_time = time.time()
        logger.info(f"=== End run_all_scenarios, total time: {overall_end_time - overall_start_time:.2f} 秒 ===")

        return {
            "exp1_baseline": df_base_stats,
            "exp2_varparams": df_sens_stats,
            "exp3_vardiscount2": df_high_stats,
            "exp4_vardiscount0": df_low_stats
        }
