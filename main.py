from database.data2json import DataProcessor
from models.HMMDataLoader import HMMDataLoader
from models.HMM import HMM
from models.run_senHMMCon_204 import SensitivityAnalyzer
from application.table_visual import table_visual
from application.fig_visual import fig_visual
from application.all_results_append import all_results_append
import pandas as pd
import yaml
from joblib import Parallel, delayed

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

disease_code=['Chronic respiratory diseases']
all_country_code=['AFG', 'AGO', 'ALB', 'AND', 'ARE', 'ARG', 'ARM', 'ASM','AUS', 'AUT', 'AZE', 'BDI', 'BEL', 'BEN', 
                  'BFA', 'BGD', 'BGR', 'BHR', 'BHS', 'BIH', 'BLR', 'BLZ', 'BMU', 'BOL', 'BRA', 'BRB', 'BRN', 'BTN', 'BWA', 'CAF',
                    'CAN', 'CHE', 'CHL', 'CHN', 'CIV', 'CMR', 'COD', 'COG', 'COK', 'COL', 'COM', 'CPV', 'CRI', 'CUB', 'CYP', 'CZE', 
                    'DEU', 'DJI', 'DNK', 'DOM', 'DZA', 'ECU', 'EGY', 'ERI', 'ESP', 'EST', 'ETH', 'FIN', 'FJI', 'FRA', 'FSM', 'GAB', 'GBR', 'GEO', 'GHA', 'GIN', 'GMB', 'GNB', 'GNQ', 'GRC', 'GRD', 'GRL', 'GTM', 'GUM', 'GUY', 
                    'HND', 'HRV', 'HTI', 'HUN', 'IDN', 'IND', 'IRL', 'IRN', 'IRQ', 'ISL', 'ISR', 'ITA', 'JAM', 'JOR', 'JPN', 'KAZ', 
                    'KEN', 'KGZ', 'KHM', 'KIR', 'KNA', 'KOR', 'KWT', 'LAO', 'LBN', 'LBR', 'LBY', 'LCA', 'LKA', 'LSO', 'LTU', 'LUX', 
                    'LVA', 'MAR', 'MCO', 'MDA', 'MDG', 'MDV', 'MEX', 'MHL', 'MKD', 'MLI', 'MLT', 'MMR', 'MNE', 'MNG', 'MNP', 'MOZ', 
                    'MRT', 'MUS', 'MWI', 'MYS', 'NAM', 'NER', 'NGA', 'NIC', 'NIU', 'NLD', 'NOR', 'NPL', 'NRU', 'NZL', 'OMN', 'PAK',
                      'PAN', 'PER', 'PHL', 'PLW', 'PNG', 'POL', 'PRI', 'PRK', 'PRT', 'PRY', 'PSE', 'QAT', 'ROU', 'RUS', 'RWA', 'SAU', 
                      'SDN', 'SEN', 'SGP', 'SLB', 'SLE', 'SLV', 'SMR', 'SOM', 'SRB', 'SSD', 'STP', 'SUR', 'SVK', 'SVN', 'SWE', 'SWZ', 
                       'SYR', 'TCD', 'TGO', 'THA', 'TJK', 'TKL', 'TKM', 'TLS', 'TON', 'TTO', 'TUN', 'TUR', 'TUV', 'TWN', 'TZA', 'UGA', 'UKR', 'URY', 'USA', 'UZB', 'VCT', 'VEN', 'VIR', 'VNM', 'VUT', 'WSM', 'YEM', 'ZAF', 'ZMB', 'ZWE']

def process_country(country_code,disease):
    config_file = './config_yml/'+disease+'.yml'
    config = read_yaml(config_file)
    processor = DataProcessor(country_code=country_code, 
                              disease=disease,
                              configname=disease,
                              startyear=config['data_layer']['startyear'],
                              projectStartYear=config['data_layer']['projectStartYear'],
                              endyear=config['data_layer']['endyear']+1,config=config)
    processor.run_full_process()
    return None 

def parallel_process(all_country_code,disease, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(delayed(process_country)(i,disease) for i in all_country_code)
   
    return results 
    
for disease in disease_code:
    parallel_process(all_country_code,disease)

for i in disease_code:
    config_file = './config_yml/'+i+'.yml'
    config = read_yaml(config_file)
    processor1 = SensitivityAnalyzer(json_dir=config['algorithm_layer']['json_dir']+i, 
                                    output_folder=config['algorithm_layer']['output_folder']+i, 
                                    n_runs=config['algorithm_layer']['n_runs'],config=config)
    processor1.process_and_merge_json_results()

   
    update=all_results_append(i)
    update.visual_update()
    processor2 = table_visual('./output/'+i+'/all_results_update.csv',
                                './database/data/dl1_countrycodeorg_country_name.csv',
                                output_dir='output/'+i, yearstart=2025,year=2050,thousand_format=True)
    processor2.run_full_process()

    loader_dir = './output/'+i+'/all_results_update.csv'
    country_region_map_path='./database/data/dl1_countrycodeorg_country_name.csv'
    fig_generator = fig_visual(loader_dir, country_region_map_path,year=2050,
                            output_dir='output/'+i)
    fig_generator.run_full_process()

