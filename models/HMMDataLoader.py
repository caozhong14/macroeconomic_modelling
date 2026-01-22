import json
import pandas as pd
import os

class HMMDataLoader:
    def __init__(self, file_path):
        """
        file_path: JSON 文件路径
        """
        self.file_path = file_path
        self.data = None
        self.df = None 
        self.load_data()
        self.to_dataframe()

    def load_data(self):
        """读取JSON文件"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def to_dataframe(self):
        """
        将JSON解析成一个DataFrame，包含以下列:
        year, sex, age_group, Population, LaborRate, EducationYear, Deaths,
        Epsilon, GDP, HealthExpenditure, SavingRate
        """
        if self.data is None:
            raise ValueError("尚未加载数据，请先调用 load_data()。")

        rows = []
        capital_share = self.data.get("CapitalShare")
        capital=self.data.get("InitialCapitalStock")
        treatment_fraction = self.data.get("TreatmentFraction")
        for year_key, year_data in self.data.items():
           
            if not year_key.isdigit():
                continue

           
            gdp = year_data.get("GDP")
            health_expenditure = year_data.get("HealthExpenditure")
            saving_rate = year_data.get("SavingRate")

           
            for sex in ["F", "M"]:
                if sex not in year_data:
                   
                    continue

                sex_data = year_data[sex]
               
                for age_key, age_value in sex_data.items():
                    row = {
                        "year": int(year_key),
                        "sex": sex,
                        "age_group": age_key,
                        "Population": age_value.get("Population"),
                        "LaborRate": age_value.get("LaborRate"),
                        "EducationYear": age_value.get("EducationYear"),
                        "Deaths_val": age_value["Deaths"].get("val"),
                        "Deaths_lower": age_value["Deaths"].get("lower"),
                        "Deaths_upper": age_value["Deaths"].get("upper"),
                        "Morbidity_lower": age_value["Morbidity"].get("lower"),
                        "Morbidity_upper": age_value["Morbidity"].get("upper"),
                        "Morbidity_val": age_value["Morbidity"].get("val"),
                        "DALYs": age_value.get("DALYs"),
                        "Epsilon": age_value.get("Epsilon"),
                        "GDP": gdp ,
                        "HealthExpenditure": health_expenditure,
                        "SavingRate": saving_rate / 100,
                        "CapitalShare": capital_share,
                        "TreatmentFraction": treatment_fraction,
                       
                        "Capital": capital
                    }
                    rows.append(row)

        self.df = pd.DataFrame(rows)
       
        self.df['age_group'] = self.df['age_group'].replace({'d00': 'd0', 'd05': 'd5'})
       
        self.df['SavingRate'] = self.df['SavingRate'].apply(lambda x: max(x, 0))


    def get_dataframe(self):
        """
        如果尚未生成DataFrame则先生成，然后返回。
        """
        if self.df is None:
            self.to_dataframe()
        return self.df
