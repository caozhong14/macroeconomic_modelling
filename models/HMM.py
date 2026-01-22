import numpy as np 
import pandas as pd
from itertools import accumulate
import statsmodels.api as sm
class HMM:
    def __init__(self, df_sq,df_cf=None, params=None,config=None):
        self.df_sq = df_sq
        self.df_cf = df_cf
        self.params = config
       
        if params is not None:
            for k, v in params.items():
                self.params['algorithm_layer'][k] = v
       
        age_groups_to_cal = [
            {'sex': 'F', 'age_group': 'd15'},
            {'sex': 'F', 'age_group': 'd20'},
            {'sex': 'F', 'age_group': 'd25'},
            {'sex': 'F', 'age_group': 'd30'},
            {'sex': 'F', 'age_group': 'd35'},
            {'sex': 'F', 'age_group': 'd40'},
            {'sex': 'F', 'age_group': 'd45'},
            {'sex': 'F', 'age_group': 'd50'},
            {'sex': 'F', 'age_group': 'd55'},
            {'sex': 'F', 'age_group': 'd60'},
            {'sex': 'F', 'age_group': 'd65'},
           
           
           
           
           
           
            {'sex': 'M', 'age_group': 'd15'},
            {'sex': 'M', 'age_group': 'd20'},
            {'sex': 'M', 'age_group': 'd25'},
            {'sex': 'M', 'age_group': 'd30'},
            {'sex': 'M', 'age_group': 'd35'},
            {'sex': 'M', 'age_group': 'd40'},
            {'sex': 'M', 'age_group': 'd45'},
            {'sex': 'M', 'age_group': 'd50'},
            {'sex': 'M', 'age_group': 'd55'},
            {'sex': 'M', 'age_group': 'd60'},
            {'sex': 'M', 'age_group': 'd65'},
           
           
           
           
           
           
        ]
        self.df_ages = pd.DataFrame(age_groups_to_cal)
        self.df_ages["averageages"] = self.df_ages['age_group'].apply(lambda x: int(x.replace("d", "")) + 2.5)
        self.df_ages.set_index(['sex', 'age_group'], inplace=True)
       
        self.cal_gain()
        
   
    def _extract_df(self):
        
        def pivot_for(colname):
            return self.df_sq.pivot_table(
                index=['sex','age_group'],
                columns='year',
                values=colname
                )
       
        def series_for(colname):
            return self.df_sq.pivot_table(
                index=['sex','age_group'],
                columns='year',
                values=colname
                ).mean(axis=0)

       
        self.df_POP = pivot_for('Population')      
        self.df_LaborRate = pivot_for('LaborRate') 
        self.df_Eduy = pivot_for('EducationYear') 
        self.df_Deaths = pivot_for('Deaths_val') 

        self.df_Morbidity = pivot_for('Morbidity_val') 

       

       

        self.sf_GDP = series_for('GDP')             
        self.sf_HealthExp = series_for('HealthExpenditure')
        self.sf_SV = series_for('SavingRate')
        self.sf_CS = series_for('CapitalShare')
        self.sf_Cap = series_for('Capital')
        self.sf_TF = series_for('TreatmentFraction')
    
   
    def _preprocess_df(self):
       
        self.DeltaDeaths = self.df_Deaths * self.params['algorithm_layer']['rho1']
        self.DeltaMorbidity = self.df_Morbidity * self.params['algorithm_layer']['rho2']
   
    def _compute_status_quo(self):
        df_ages_expand = self.df_Eduy.copy()
        for column in df_ages_expand.columns:
            df_ages_expand[column] = self.df_ages

        def mincer(eduy, ages, eta1=self.params['algorithm_layer']['eta1'], 
                   eta2=self.params['algorithm_layer']['eta2'], eta3=self.params['algorithm_layer']['eta3']):
            x = ages - eduy - 5 
            h = eta1 * eduy + eta2 * x + eta3 * x * x
            capital_pc = np.exp(h)
            return capital_pc
        self.labor = self.df_POP * self.df_LaborRate * mincer(self.df_Eduy, df_ages_expand)
        self.L = self.labor.sum(axis=0)
       
        init_capital = self.sf_Cap.iloc[0]
        cd = self.params['algorithm_layer']['cd']
       
        f = self.sf_SV * self.sf_GDP
       
        capital = [(1 - cd) * init_capital + f.iloc[0]]
       
        for i in range(1, len(self.sf_GDP)):
            capital.append((1 - cd) * capital[-1] + f.iloc[i])
       
        self.K = pd.Series(capital, index=self.L.index)   

       
        self.Y_sq = self.sf_GDP
       
        self.At = self.sf_GDP / (self.K**self.sf_CS * (self.L ** (1 - self.sf_CS)))
       
       
       
       
       

       
       
       

       
       

       
       
       

       
       
       
       

       
       
       

       
       
       

       
       

       
       
       
       

       
       
       
       
       
       
       
       
       


   
    def _compute_counterfactual(self):
        """
        计算反事实场景：考虑死亡率和患病率对劳动力的冲击。
        """
       
        df_ages_expand = self.df_Eduy.copy()
       
        for column in df_ages_expand.columns:
           
            df_ages_expand[column] = self.df_ages

        def mincer(eduy, ages, rho=self.params['algorithm_layer']['eta1'], 
                   beta1=self.params['algorithm_layer']['eta2'], beta2=self.params['algorithm_layer']['eta3']):
            """
            计算人均资本。
            
            参数:
            eduy (float): 教育年限
            ages (float): 年龄
            rho (float): 教育回报率
            beta1 (float): 第一个系数
            beta2 (float): 第二个系数
            
            返回:
            float: 人均资本
            """
           
            x = ages - eduy - 5 
           
            h = rho * eduy + beta1 * x + beta2 * x * x
           
            capital_pc = np.exp(h)
            return capital_pc

        def DeltaRatioDeath(sex, age_group, year):
            """
            针对特定性别和年龄组，计算死亡率的影响比率。
            
            参数:
            sex (str): 性别
            age_group (str): 年龄组
            year (int): 年份
            
            返回:
            float: 死亡率的影响比率
            """
           
            start_year = int(self.DeltaDeaths.columns[0])
           
           
            time = year - start_year + 1
           
            age = int(age_group.replace("d", "")) + 2
           
            t = min(age, time)
           
            temp = 1.0
           
            for i in range(0, t):
               
                impact_age = age - i - 1
               
                impact_time = time - i - 1 
               
                if impact_age >= 0 and impact_time >= 0:
                   
                    impact_age_group = 'd' + str(impact_age // 5 * 5)
                    temp *= (1 - self.DeltaDeaths.loc[sex].loc[impact_age_group][start_year+impact_time])
           
            ratio = 1.0 / temp 
            return ratio

        def DeltaRatioMorbid(sex, age_group, year):
            """
            针对特定性别和年龄组，计算患病率的影响比率。
            
            参数:
            sex (str): 性别
            age_group (str): 年龄组
            year (int): 年份
            
            返回:
            float: 患病率的影响比率
            """
           
            start_year = int(self.DeltaDeaths.columns[0])
           
            time = year - start_year + 1
           
            age = int(age_group.replace("d", "")) + 2
           
            t = min(age, time)
           
            temp = 1.0
           
            for i in range(0, t):
               
                impact_age = age - i - 1
               
                impact_time = time - i - 1 
               
                if impact_age >= 0 and impact_time >= 0:
                   
                    impact_age_group = 'd' + str(impact_age // 5 * 5)
                    temp *= (1 - self.params['algorithm_layer']['pc'] ** i * self.df_Morbidity.loc[sex].loc[impact_age_group][start_year+impact_time])
           
            ratio = 1.0 / temp 
            return ratio

       
        self.df_POP_cf, self.df_LaborRate_cf = self.df_POP.copy(), self.df_LaborRate.copy()
       
       
       
       
       
       
       
       
        growth_ratios = pd.DataFrame(
            {
                year: self.df_POP_cf.index.map(lambda idx: DeltaRatioDeath(idx[0], idx[1], year))
                for year in self.df_POP_cf.columns
            },
            index=self.df_POP_cf.index
        )
        self.df_POP_cf = self.df_POP_cf * growth_ratios

       
       
       
       
       
       
        labor_growth = pd.DataFrame({
            year: self.df_LaborRate_cf.index.map(lambda idx: DeltaRatioMorbid(idx[0], idx[1], year))
            for year in self.df_LaborRate_cf.columns
        }, index=self.df_LaborRate_cf.index)
        self.df_LaborRate_cf = self.df_LaborRate_cf * labor_growth

       
        self.labor_cf = self.df_POP_cf * self.df_LaborRate_cf * mincer(self.df_Eduy, df_ages_expand)
       
        self.L_cf = self.labor_cf.sum(axis=0)
       
        init_capital = self.sf_Cap.iloc[0]
       
        self.POP = self.df_POP_cf.sum(axis=0)
       
        cd = self.params['algorithm_layer']['cd']
       
        f = self.sf_SV * (self.sf_GDP + self.POP * self.sf_HealthExp * self.sf_TF)
       
        capital = [(1 - cd) * init_capital + f.iloc[0]]
       
        for i in range(1, len(self.sf_GDP)):
            capital.append((1 - cd) * capital[-1] + f.iloc[i])
       
        self.K_cf = pd.Series(capital, index=self.L_cf.index)                
       
        self.Y_cf = self.At * self.K_cf**self.sf_CS * (self.L_cf ** (1 - self.sf_CS))


    def apply_discount(self, discount=None):
        if discount is not None:
            assert (discount >= 0) and (discount < 0.1)
            self.params['algorithm_layer']['ds'] = discount
       
        self.GDPloss = self.Y_cf - self.Y_sq

       
       
       
       
       
       
       

        discounts = np.array([(1 - self.params['algorithm_layer']['ds']) ** i for i in range(len(self.GDPloss))])
        self.GDPlossDiscount = self.GDPloss * discounts
       
        self.GDPlossDiscount = list(accumulate(self.GDPlossDiscount))
        self.GDPDiscount = self.sf_GDP * discounts
        self.GDPDiscount=list(accumulate(self.GDPDiscount))
        self.GDPShare = pd.Series(self.GDPlossDiscount)/ pd.Series(self.GDPDiscount)
       
        self.GDPper=self.GDPlossDiscount/self.df_POP.sum(axis = 0)
       
        self.L_contri=(1-self.sf_CS)*np.log(self.L_cf/self.L)/np.log(self.Y_cf/self.Y_sq)
        self.K_contri=self.sf_CS*np.log(self.K_cf/self.K)/np.log(self.Y_cf/self.Y_sq)
       
       
        self.df_result = pd.DataFrame({
            'year': self.L_contri.index,
            'GDPloss': self.GDPlossDiscount,
            'Lcontri':self.L_contri,
            'Kcontri':self.K_contri,
            'GDPper':self.GDPper,
            'GDPShare':self.GDPShare.values
        })
    def cal_gain(self):
       
        self._extract_df()
        self._preprocess_df()
        self._compute_status_quo()
        self._compute_counterfactual()
        self.apply_discount()

