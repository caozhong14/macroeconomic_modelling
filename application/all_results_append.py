import pandas as pd
import os
import numpy as np
from models.HMMDataLoader import HMMDataLoader
from itertools import accumulate




class all_results_append:
    def __init__(self,disease):
        self.disease=disease
    def visual_update(self):
        directory = f'./database/json_output/{self.disease}/'
        all_data = pd.DataFrame()
        
       
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                country_code = filename[:-5] 
                loader = HMMDataLoader(os.path.join(directory, filename))
                df_country = loader.get_dataframe()
               
               
               
                filtered_df = df_country.copy() 
               
                result = filtered_df.groupby('year').apply(
                    lambda x: pd.Series({
                        'POP': x['Population'].sum(),
                        'Mortality': (x['Deaths_val'] * x['Population']).sum() / x['Population'].sum(),
                        'Morbidity': (x['Morbidity_val'] * x['Population']).sum() / x['Population'].sum(),
                        'DALYs': (x['DALYs'] * x['Population']).sum() / x['Population'].sum(),
                        'GDP': x['GDP'].mean() 
                    })
                ).reset_index()
                
                result['country_codes'] = country_code
                all_data = pd.concat([all_data, result], ignore_index=True)
        
       
        all_data = all_data.sort_values(['country_codes', 'year'])
        all_data['GDP_cum_exp1'] = np.nan 
        all_data['GDP_cum_exp3'] = np.nan 
        all_data['GDP_cum_exp4'] = np.nan 
        
        for country_code in all_data['country_codes'].unique():
            mask = all_data['country_codes'] == country_code
            gdp_series = all_data.loc[mask, 'GDP'].values
            if len(gdp_series) == 0:
                continue
            
           
            discounts = np.array([(1 - 0.03) ** i for i in range(len(gdp_series))])
           
            cum_gdp = np.cumsum(gdp_series * discounts)
            all_data.loc[mask, 'GDP_cum_exp1'] = cum_gdp
            
           
            discounts = np.array([(1 - 0.02) ** i for i in range(len(gdp_series))])
           
            cum_gdp = np.cumsum(gdp_series * discounts)
            all_data.loc[mask, 'GDP_cum_exp3'] = cum_gdp
            
           
            discounts = np.array([(1 - 0) ** i for i in range(len(gdp_series))])
           
            cum_gdp = np.cumsum(gdp_series * discounts)
            all_data.loc[mask, 'GDP_cum_exp4'] = cum_gdp
       
        template_df = pd.read_csv('all_results_template.csv')[['country_codes', 'task', 'year']]
        result_df=pd.read_csv(f'./output/{self.disease}/all_results.csv')
        final_data = pd.merge(template_df, all_data, on=['country_codes', 'year'], how='left')
        
       
        conditions = [
            final_data['task'] == 'exp1_baseline',
           
            final_data['task'] == 'exp3_vardiscount2',
            final_data['task'] == 'exp4_vardiscount0'
        ]
        choices = [
            final_data['GDP_cum_exp1'],
           
            final_data['GDP_cum_exp3'],
            final_data['GDP_cum_exp4']
        ]
        final_data['GDP'] = np.select(conditions, choices, default=final_data['GDP'])
        final_data.drop(['GDP_cum_exp1', 'GDP_cum_exp3', 'GDP_cum_exp4'], axis=1, inplace=True)
        
        final_data_merge=pd.merge(result_df, final_data, on=['country_codes','task', 'year'], how='left')
       
        mask_exp2_varparams = final_data_merge['task'] == 'exp2_varparams'
       
       
        final_data_merge.loc[mask_exp2_varparams, 'GDP'] = \
            final_data_merge.loc[mask_exp2_varparams, 'GDPloss_val'] / \
            final_data_merge.loc[mask_exp2_varparams, 'GDPShare_val']
        
       
        final_data_merge['POP_cum'] = final_data_merge.groupby(['country_codes', 'task'])['POP'].cumsum()
        final_data_merge['year_count'] = final_data_merge.groupby(['country_codes', 'task']).cumcount() + 1
        final_data_merge['POP_annual'] = final_data_merge['POP_cum'] / final_data_merge['year_count']

       
        final_data_merge = final_data_merge.drop(['GDPper_lower', 'GDPper_val', 'GDPper_upper'], axis=1, errors='ignore')
        final_data_merge['GDPper_lower'] = final_data_merge['GDPloss_lower'] / final_data_merge['POP_annual']
        final_data_merge['GDPper_val']   = final_data_merge['GDPloss_val']   / final_data_merge['POP_annual']
        final_data_merge['GDPper_upper'] = final_data_merge['GDPloss_upper'] / final_data_merge['POP_annual']

       
        output_dir = f'./output/{self.disease}'
        os.makedirs(output_dir, exist_ok=True)
        final_data_merge.to_csv(os.path.join(output_dir, 'all_results_update.csv'), index=False)
