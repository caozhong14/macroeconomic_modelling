import plotly.graph_objects as go
import pandas as pd
import plotly.io as pio
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class fig_visual:
    def __init__(self, loader_dir, country_region_map_path, year, output_dir='./output'):
        self.file_path = loader_dir
        self.country_region_map_path = country_region_map_path
        
        self.output_dir = output_dir  
        os.makedirs(self.output_dir, exist_ok=True)  
        self.country_region_map, self.region_map, self.income_group_map = self._load_country_region_map()
        self.default_layout = {
            'font': dict(family="Times New Roman", size=10),
            'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
            'geo': dict(
                showframe=False,
                showcoastlines=False,
                showcountries=True,
                countrycolor="rgb(0, 0, 0)",
                countrywidth=0.1,
                projection_type='equirectangular'
            )
        }
            
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        log_file_path = os.path.join(self.output_dir, 'figure_logfile.log')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _load_country_region_map(self):
        """加载国家代码到国家名称的映射"""
        try:
            country_region_df = pd.read_csv(self.country_region_map_path, encoding='latin1')
            country_region_map = dict(zip(country_region_df['Country Code'], country_region_df['country']))
            region_map = country_region_df.set_index('country')['Region'].to_dict()
            income_group_map = country_region_df.set_index('country')['Income group'].to_dict()
            return country_region_map, region_map, income_group_map
        except Exception as e:
            raise ValueError(f"无法加载国家区域映射文件: {e}")

    def _validate_data(self):
        """验证数据格式是否符合要求"""
        if 'year' not in self.raw_df.columns:
            raise ValueError("year 列缺失")
        if 'country_codes' not in self.raw_df.columns:
            raise ValueError("country_codes 列缺失")

    def _load_data(self,year=2050):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.raw_df = pd.read_csv(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"文件未找到: {self.file_path}")
        except Exception as e:
            raise Exception(f"读取文件时发生错误: {e}")

        self._validate_data()
        self.raw_df = self.raw_df[self.raw_df['year'] == year]
        self.raw_df = self.raw_df[self.raw_df['task'] == "exp1_baseline"]
        self.raw_df = self.raw_df.drop(columns=['year','task'])
        return self.raw_df

    def generate_choropleth(self, z_column, colorscale, colorbar_title,
                           reversescale=False, z_transform=None, colorbar_kwargs=None):
        

        
        
        
        
        if z_transform:
            z_data = self.df[z_column].apply(z_transform)
        else:
            z_data = self.df[z_column]

        
        cb_config = {
            'len': 0.4,
            'thickness': 15,
            'title': colorbar_title,
        }
        if colorbar_kwargs:
            cb_config.update(colorbar_kwargs)
            if 'tickvals' in colorbar_kwargs:
                cb_config.setdefault('tickmode', 'array')

        
        fig = go.Figure(go.Choropleth(
            locations=self.df['country_codes'],
            z=z_data,
            colorscale=colorscale,
            autocolorscale=False,
            reversescale=reversescale,
            marker=dict(
                line=dict(color='rgb(180,180,180)', width=0.2)
            ),
            colorbar=cb_config
        ))

        
        fig.update_layout(**self.default_layout)

        
        if fig.data:
            fig.data[0].colorbar.x = 0.0
            fig.data[0].colorbar.y = 0.4
        
        
        
        
        
        
        return fig

    def fig1_visual(self):
        try:
            self.df = self._load_data()
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
       
        self.df['GDPloss_val_billion'] = self.df['GDPloss_val'] / 1e9 
        self.df['GDPloss_val_log'] = self.df['GDPloss_val_billion'].apply(np.log10)
        fig1 = self.generate_choropleth(
            z_column='GDPloss_val_log',
            colorscale=[[0, "rgb(5, 10, 172)"], [0.35, "rgb(40, 60, 190)"], [1, "rgb(210, 240, 255)"]],
            colorbar_title='Macroeconomic burden<br>from 2025 to 2050<br>(in billions of 2024 U.S.$)',
            reversescale=True,
            colorbar_kwargs={
                'tickvals': [0,1,2,3],
                'ticktext': ['$1', '$10', '$100', '$1000'],
                'tickmode': 'array'
            }
        )
       

        fig1.write_html(os.path.join(self.output_dir, 'fig1.html'))
       
        fig1.write_image(os.path.join(self.output_dir, 'fig1.pdf'))
        fig1.write_image(os.path.join(self.output_dir, 'fig1.png'), scale=3)

    def fig2_visual(self):
        try:
            self.df = self._load_data()
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
        fig2 = self.generate_choropleth(
            z_column='GDPShare_val',
            colorscale=[[0, "rgb(5, 10, 172)"], [0.35, "rgb(40, 60, 190)"], [1, "rgb(210, 240, 255)"]],
            colorbar_title='Macroeconomic burden<br>in % of total GDP <br>from 2025 to 2050',
            reversescale=True,
            z_transform=lambda x: x * 100,
           
        )

        fig2.write_html(os.path.join(self.output_dir, 'fig2.html'))
        fig2.write_image(os.path.join(self.output_dir, 'fig2.pdf'))
    
    def fig9_visual(self):
        try:
            self.df = self._load_data()
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
        fig9 = self.generate_choropleth(
            z_column='Mortality',
            colorscale=[[0, "rgb(5, 10, 172)"], [0.35, "rgb(40, 60, 190)"], [1, "rgb(210, 240, 255)"]],
            colorbar_title='Mortality per 1000',
            reversescale=True,
            z_transform=lambda x: x * 1000,
           
        )

        fig9.write_html(os.path.join(self.output_dir, 'fig9.html'))
        fig9.write_image(os.path.join(self.output_dir, 'fig9.pdf'))
    
    def fig11_visual(self):
        try:
            self.df = self._load_data(year=2025)
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
        fig11 = self.generate_choropleth(
            z_column='Mortality',
            colorscale=[[0, "rgb(5, 10, 172)"], [0.35, "rgb(40, 60, 190)"], [1, "rgb(210, 240, 255)"]],
            colorbar_title='Mortality per 100,000',
            reversescale=True,
            z_transform=lambda x: x * 100000,
           
        )

        fig11.write_html(os.path.join(self.output_dir, 'fig11.html'))
        fig11.write_image(os.path.join(self.output_dir, 'fig11.pdf'))

    def fig10_visual(self):
        try:
            self.df = self._load_data()
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
        fig10 = self.generate_choropleth(
            z_column='Morbidity',
            colorscale=[[0, "rgb(5, 10, 172)"], [0.35, "rgb(40, 60, 190)"], [1, "rgb(210, 240, 255)"]],
            colorbar_title='Morbidity per 1,000',
            reversescale=True,
            z_transform=lambda x: x * 1000,
           
        )

        fig10.write_html(os.path.join(self.output_dir, 'fig10.html'))
        fig10.write_image(os.path.join(self.output_dir, 'fig10.pdf'))
    
    def fig12_visual(self):
        try:
            self.df = self._load_data(year=2025)
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
        fig12 = self.generate_choropleth(
            z_column='Morbidity',
            colorscale=[[0, "rgb(5, 10, 172)"], [0.35, "rgb(40, 60, 190)"], [1, "rgb(210, 240, 255)"]],
            colorbar_title='Morbidity per 1,000',
            reversescale=True,
            z_transform=lambda x: x * 1000,
           
        )

        fig12.write_html(os.path.join(self.output_dir, 'fig12.html'))
        fig12.write_image(os.path.join(self.output_dir, 'fig12.pdf'))


    def fig3_visual(self):
        try:
            self.df = self._load_data()
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None

       
        self.df['country'] = self.df['country_codes'].map(self.country_region_map)
        self.df['region'] = self.df['country'].map(self.region_map)

       
        region_data = self.df.groupby('region')['Lcontri_val'].mean().reset_index()
        region_data['b'] = 1 - region_data['Lcontri_val']
        region_data = region_data.sort_values(by='Lcontri_val', ascending=False)

       
        fig, ax = plt.subplots(figsize=(10, 6))

       
        color_a = 'skyblue'
        color_b = 'lightcoral'

       
        ax.bar(region_data['region'], region_data['Lcontri_val'], label='Labor Contribution', color=color_a)
        ax.bar(region_data['region'], region_data['b'], bottom=region_data['Lcontri_val'], label='Capital Contribution', color=color_b)

       
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
       
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
       
        plt.xticks(rotation=45, ha='right')

       
        plt.tight_layout()

       
        output_path = os.path.join(self.output_dir, 'fig3.pdf')
        plt.savefig(output_path, format='pdf')
        plt.close(fig) 
        print(f"Region bar plot saved to {output_path}")

    def fig4_visual(self):
        try:
            self.df = self._load_data()
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None

       
        self.df['country'] = self.df['country_codes'].map(self.country_region_map)
        self.df['income_group'] = self.df['country'].map(self.income_group_map)

       
        income_group_data = self.df.groupby('income_group')['Lcontri_val'].mean().reset_index()
        income_group_data['b'] = 1 - income_group_data['Lcontri_val']
        income_group_data = income_group_data.sort_values(by='Lcontri_val', ascending=False)

       
        fig, ax = plt.subplots(figsize=(10, 6))

       
        color_a = 'skyblue'
        color_b = 'lightcoral'

       
        ax.bar(income_group_data['income_group'], income_group_data['Lcontri_val'], label='Labor Contribution', color=color_a)
        ax.bar(income_group_data['income_group'], income_group_data['b'], bottom=income_group_data['Lcontri_val'], label='Capital Contribution', color=color_b)

       
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.legend()
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
       
        plt.xticks(rotation=45, ha='right')

       
        plt.tight_layout()

       
        output_path = os.path.join(self.output_dir, 'fig4.pdf')
        plt.savefig(output_path, format='pdf')
        plt.close(fig) 
        print(f"Income Group bar plot saved to {output_path}")

    def fig5_visual(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.raw_df = pd.read_csv(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"文件未找到: {self.file_path}")
        except Exception as e:
            raise Exception(f"读取文件时发生错误: {e}")

        self._validate_data()
        self.raw_df = self.raw_df[self.raw_df['task'] == "exp1_baseline"]
        self.raw_df['country'] = self.raw_df['country_codes'].map(self.country_region_map)
        self.raw_df['region'] = self.raw_df['country'].map(self.region_map)
        region_year_data = self.raw_df.groupby(['region', 'year'])['Lcontri_val'].mean().reset_index()

       
       
       
        plt.style.use('seaborn-whitegrid') 
        plt.rcParams.update({
            'font.family': 'Arial',         
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'lines.linewidth': 1.5,        
            'lines.markersize': 6,          
            'axes.spines.right': False,     
            'axes.spines.top': False,       
            'figure.dpi': 300               
        })

       
        color_palette = [
            '#2E5984', 
            '#E69F36', 
            '#6A8D73', 
            '#C1272D', 
            '#7F7F7F'  
        ]

       
       
       
        fig, ax = plt.subplots(figsize=(8, 5))
        regions = region_year_data['region'].unique()
        
       
        for idx, region in enumerate(regions):
            region_df = region_year_data[region_year_data['region'] == region]
            ax.plot(
                region_df['year'], 
                region_df['Lcontri_val'] * 100, 
                label=region,
                color=color_palette[idx % len(color_palette)],
                marker='o',                     
                markersize=5,
                linestyle='--' if idx % 2 == 0 else '-', 
                linewidth=1.5
            )

       
       
       
        ax.set_xlabel('Year', fontweight='bold', labelpad=10)
        ax.set_ylabel('Labor Contribution (%)', fontweight='bold', labelpad=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%')) 
        
       
       
        
       
        ax.set_xticks(sorted(region_year_data['year'].unique()))
        ax.grid(False)
        ax.tick_params(axis='x', rotation=45, pad=5)
        
       
        legend = ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            frameon=False,
            title='Region',
            title_fontsize='10'
        )
        
        plt.tight_layout()
        self.logger.info("Figure5:Labor Contribution Trends by Region")
        output_path = os.path.join(self.output_dir, 'fig5.pdf')
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close(fig)

    def fig6_visual(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.raw_df = pd.read_csv(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"文件未找到: {self.file_path}")
        except Exception as e:
            raise Exception(f"读取文件时发生错误: {e}")

        self._validate_data()
        self.raw_df = self.raw_df[self.raw_df['task'] == "exp1_baseline"]
        self.raw_df['country'] = self.raw_df['country_codes'].map(self.country_region_map)
        self.raw_df['income_group'] = self.raw_df['country'].map(self.income_group_map)
        income_group_year_data = self.raw_df.groupby(['income_group', 'year'])['Lcontri_val'].mean().reset_index()

       
       
       
        plt.style.use('seaborn-whitegrid')
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'axes.prop_cycle': plt.cycler(color=['#00468B', '#ED0000', '#42B540', '#0099B4']), 
            'figure.dpi': 300
        })

       
       
       
        fig, ax = plt.subplots(figsize=(8, 5))
        income_groups = income_group_year_data['income_group'].unique()
        
       
        for idx, group in enumerate(income_groups):
            group_df = income_group_year_data[income_group_year_data['income_group'] == group]
            ax.plot(
                group_df['year'], 
                group_df['Lcontri_val'] * 100,
                label=group,
                marker='s' if idx % 2 == 0 else '^', 
                markersize=6,
                linestyle='-',
                linewidth=1.8,
                alpha=0.9
            )

       
       
       
       
        ax.spines['left'].set_color('#808080')
        ax.grid(False)
        ax.spines['bottom'].set_color('#808080')
        ax.tick_params(axis='both', colors='#404040')
        
       
        for line in ax.get_lines():
            x_data = line.get_xdata()
            y_data = line.get_ydata()
            for x, y in zip([x_data[0], x_data[-1]], [y_data[0], y_data[-1]]):
                ax.text(
                    x, y + 0.5, f'{y:.1f}%',
                    ha='center', va='bottom',
                    color=line.get_color(),
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8)
                )
        ax.set_xlabel('Year', 
                    fontsize=11, 
                    fontweight='bold',
                    color='#404040',
                    labelpad=12)
        
        ax.set_ylabel('Labor Contribution (%)', 
                    fontsize=11,
                    fontweight='bold',
                    color='#404040',
                    labelpad=12)
        
       
        legend = ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.05, 1),
            title='Income Group',
            title_fontsize='10',
            frameon=True,
            facecolor='#F5F5F5',
            edgecolor='none'
        )

        plt.tight_layout()
        self.logger.info("Figure6:Labor Contribution Trends by Income Group")
        output_path = os.path.join(self.output_dir, 'fig6.pdf')
        plt.savefig(output_path, format='pdf', bbox_inches='tight', transparent=True) 
        plt.close(fig)

    def fig7_visual(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.df = pd.read_csv(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"文件未找到: {self.file_path}")
        except Exception as e:
            raise Exception(f"读取文件时发生错误: {e}")

       
        self.df = self.df[self.df['task'] == "exp1_baseline"].copy()
        self.df['country'] = self.df['country_codes'].map(self.country_region_map)
        self.df['region'] = self.df['country'].map(self.region_map)

       
        for col in ['GDPloss_val', 'Kcontri_val', 'Lcontri_val', 'year']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df = self.df.dropna(subset=['country', 'region', 'year', 'GDPloss_val', 'Kcontri_val', 'Lcontri_val'])

       
        df_cy = self.df.groupby(['country', 'year', 'region'], as_index=False).agg({
            'GDPloss_val': 'sum',
            'Kcontri_val': 'mean',
            'Lcontri_val': 'mean'
        })

       
        def compute_country_cumulative(g):
            g = g.sort_values('year').copy()
            g['annual_burden'] = g['GDPloss_val'].diff().fillna(g['GDPloss_val'])
           
            negs = (g['annual_burden'] < 0).sum()
            if negs > 0:
                self.logger.warning(f"Found {negs} negative annual burdens for country {g['country'].iloc[0]}; set to 0.")
                g.loc[g['annual_burden'] < 0, 'annual_burden'] = 0.0
            g['yearly_capital'] = g['annual_burden'] * g['Kcontri_val']
            g['yearly_labor'] = g['annual_burden'] * g['Lcontri_val']
            g['cum_capital'] = g['yearly_capital'].cumsum()
            g['cum_labor'] = g['yearly_labor'].cumsum()
           
            denom = g['cum_capital'] + g['cum_labor']
            g['cum_cap_share'] = np.where(denom > 0, g['cum_capital'] / denom * 100.0, 0.0)
            return g

        df_cumulative = df_cy.groupby('country', group_keys=False).apply(compute_country_cumulative).reset_index(drop=True)

       
        region_agg = df_cumulative.groupby(['region', 'year'], as_index=False).agg({
            'cum_capital': 'sum',
            'cum_labor': 'sum'
        })
       
        denom = region_agg['cum_capital'] + region_agg['cum_labor']
        region_agg['avg_kcontri'] = np.where(denom > 0, region_agg['cum_capital'] / denom * 100.0, 0.0)

       
        regions = ["Global", "North America", "East Asia and Pacific", "Europe and Central Asia",
                "Latin America and Caribbean", "Middle East and North Africa", "South Asia", "Sub-Saharan Africa"]
        years = sorted(region_agg['year'].unique())
        full_index = pd.MultiIndex.from_product([regions, years], names=['region', 'year'])
        region_agg = region_agg.set_index(['region', 'year']).reindex(full_index, fill_value=0).reset_index()

       
        total_by_year = region_agg.groupby('year', as_index=False).agg({
            'cum_capital': 'sum',
            'cum_labor': 'sum'
        })
        total_by_year['region'] = 'Global'
        denom = total_by_year['cum_capital'] + total_by_year['cum_labor']
        total_by_year['avg_kcontri'] = np.where(denom > 0, total_by_year['cum_capital'] / denom * 100.0, 0.0)

       
        region_agg = region_agg[region_agg['region'] != 'Global']
        region_agg = pd.concat([pd.DataFrame(total_by_year), region_agg], ignore_index=True)

       
        region_agg = region_agg.rename(columns={'cum_capital': 'log_capital', 'cum_labor': 'log_labor'})

       
        fontsize=18
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': fontsize,
            'axes.titlesize': fontsize,
            'axes.labelsize': fontsize,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'legend.fontsize': 15,
            'figure.dpi': 600
        })

        color_palette = {
            'capital': '#2E5984', 'labor': '#E69F36',
            'line': '#C1272D', 'text': '#404040'
        }
        years = sorted(region_agg['year'].unique())

       
        selected_years = [y for y in years if y % 5 == 0]
        if not selected_years or selected_years[0] != years[0]:
            selected_years.insert(0, years[0])
        if selected_years[-1] != years[-1]:
            selected_years.append(years[-1])

        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        axes = axes.flatten()

        row1_regions = regions[:4]
        row2_regions = regions[4:]

        row1_max_burden = region_agg[region_agg['region'].isin(row1_regions)][['log_capital', 'log_labor']].sum(axis=1).max()
        row2_max_burden = region_agg[region_agg['region'].isin(row2_regions)][['log_capital', 'log_labor']].sum(axis=1).max()

        row1_y_range = (0, (row1_max_burden if not np.isnan(row1_max_burden) else 0) * 1.05)
        row2_y_range = (0, (row2_max_burden if not np.isnan(row2_max_burden) else 0) * 1.05)

        for idx, region in enumerate(regions):
            ax = axes[idx]
            ax.grid(False)
            region_df = region_agg[region_agg['region'] == region].sort_values('year')

            capital_vals = region_df['log_capital'].values
            labor_vals = region_df['log_labor'].values
            avg_k = region_df['avg_kcontri'].values

            capital_bars = ax.bar(years, capital_vals, color=color_palette['capital'], width=0.8)
            labor_bars = ax.bar(years, labor_vals, bottom=capital_vals, color=color_palette['labor'], width=0.8)

            ax2 = ax.twinx()
            ax2.grid(False)
            ax2.plot(years, avg_k, marker='o', markersize=3, linestyle='--', color=color_palette['line'], linewidth=1.5)

            ax2.spines['right'].set_color('grey')
            ax2.spines['right'].set_visible(True)
            ax2.set_ylabel('Capital Contribution (%)', color='black', labelpad=3)
            ax2.tick_params(axis='y', colors='black')

            if region in row1_regions:
                ax.set_ylim(row1_y_range)
            else:
                ax.set_ylim(row2_y_range)
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))

            ax.set_xticks(selected_years)
            ax.set_xticklabels(selected_years, rotation=45, ha='right')
            ax.set_xlabel('Year', color=color_palette['text'], labelpad=5)
            ax.set_ylabel('Economic Burden (2024 U.S. $)', color=color_palette['text'], labelpad=3)

            ax2.set_ylim(0, 100)
            ax.set_title(region, fontweight='bold', color=color_palette['text'], pad=15)

            for i, (x, y) in enumerate(zip(years, avg_k)):
                if i in [0, len(years)-1]:
                    offset = 2.5 if y < 95 else -3.5
                   
                    ax2.text(x, y + offset, f'{y:.3f}%', ha='center', color=color_palette['text'], fontsize=12)

        legend_elements = [
            plt.Rectangle((0,0),1,1, color=color_palette['capital'], label='Capital Burden'),
            plt.Rectangle((0,0),1,1, color=color_palette['labor'], label='Labor Burden'),
            plt.Line2D([0],[0], marker='o', color=color_palette['line'], label='Capital Contribution', linestyle='--', markersize=5)
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=3, frameon=False)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88, hspace=0.4, wspace=0.3)
        self.logger.info("Figure7: Trends in cumulative Economic Burden and Capital Contribution by World Bank region")
        plt.savefig(os.path.join(self.output_dir, 'fig7.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)


    def fig8_visual(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.df = pd.read_csv(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"文件未找到: {self.file_path}")
        except Exception as e:
            raise Exception(f"读取文件时发生错误: {e}")

       
        self.df = self.df[self.df['task'] == "exp1_baseline"].copy()
        self.df['country'] = self.df['country_codes'].map(self.country_region_map)
        self.df['income_group'] = self.df['country'].map(self.income_group_map)

       
        for col in ['GDPloss_val', 'Kcontri_val', 'Lcontri_val', 'year']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df = self.df.dropna(subset=['country', 'income_group', 'year', 'GDPloss_val', 'Kcontri_val', 'Lcontri_val'])

       
        df_cy = self.df.groupby(['country', 'year', 'income_group'], as_index=False).agg({
            'GDPloss_val': 'sum',
            'Kcontri_val': 'mean',
            'Lcontri_val': 'mean'
        })

       
        def compute_country_cumulative(g):
            g = g.sort_values('year').copy()
            g['annual_burden'] = g['GDPloss_val'].diff().fillna(g['GDPloss_val'])
            negs = (g['annual_burden'] < 0).sum()
            if negs > 0:
                self.logger.warning(f"Found {negs} negative annual burdens for country {g['country'].iloc[0]}; set to 0.")
                g.loc[g['annual_burden'] < 0, 'annual_burden'] = 0.0
            g['yearly_capital'] = g['annual_burden'] * g['Kcontri_val']
            g['yearly_labor'] = g['annual_burden'] * g['Lcontri_val']
            g['cum_capital'] = g['yearly_capital'].cumsum()
            g['cum_labor'] = g['yearly_labor'].cumsum()
            denom = g['cum_capital'] + g['cum_labor']
            g['cum_cap_share'] = np.where(denom > 0, g['cum_capital'] / denom * 100.0, 0.0)
            return g

        df_cumulative = df_cy.groupby('country', group_keys=False).apply(compute_country_cumulative).reset_index(drop=True)

       
        income_agg = df_cumulative.groupby(['income_group', 'year'], as_index=False).agg({
            'cum_capital': 'sum',
            'cum_labor': 'sum'
        })
        denom = income_agg['cum_capital'] + income_agg['cum_labor']
        income_agg['avg_kcontri'] = np.where(denom > 0, income_agg['cum_capital'] / denom * 100.0, 0.0)

       
        income_groups = ["High income", "Upper middle income", "Lower middle income", "Low income"]
        years = sorted(income_agg['year'].unique())
        full_index = pd.MultiIndex.from_product([income_groups, years], names=['income_group', 'year'])
        income_agg = income_agg.set_index(['income_group', 'year']).reindex(full_index, fill_value=0).reset_index()

       
        income_agg = income_agg.rename(columns={'cum_capital': 'log_capital', 'cum_labor': 'log_labor'})

       
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'font.size': 15,
            'axes.titlesize': 15,
            'axes.labelsize': 15,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.dpi': 600
        })

        color_palette = {
            'capital': '#2E5984', 'labor': '#E69F36',
            'line': '#C1272D', 'text': '#404040'
        }
        years = sorted(income_agg['year'].unique())

        selected_years = [y for y in years if y % 5 == 0]
        if not selected_years or selected_years[0] != years[0]:
            selected_years.insert(0, years[0])
        if selected_years[-1] != years[-1]:
            selected_years.append(years[-1])

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        row1_groups = income_groups[:2]
        row2_groups = income_groups[2:]

        row1_max_burden = income_agg[income_agg['income_group'].isin(row1_groups)][['log_capital', 'log_labor']].sum(axis=1).max()
        lower_middle_burden = income_agg[income_agg['income_group'] == "Lower middle income"][['log_capital', 'log_labor']].sum(axis=1).max()
        low_income_burden = income_agg[income_agg['income_group'] == "Low income"][['log_capital', 'log_labor']].sum(axis=1).max()

        row1_y_range = (0, (row1_max_burden if not np.isnan(row1_max_burden) else 0) * 1.05)
        lower_middle_y_range = (0, (lower_middle_burden if not np.isnan(lower_middle_burden) else 0) * 1.05)
        low_income_y_range = (0, (low_income_burden if not np.isnan(low_income_burden) else 0) * 2)

        for idx, income_group in enumerate(income_groups):
            ax = axes[idx]
            ax.grid(False)
            group_df = income_agg[income_agg['income_group'] == income_group].sort_values('year')

            capital_vals = group_df['log_capital'].values
            labor_vals = group_df['log_labor'].values
            avg_k = group_df['avg_kcontri'].values

            ax.bar(years, capital_vals, color=color_palette['capital'], width=0.8)
            ax.bar(years, labor_vals, bottom=capital_vals, color=color_palette['labor'], width=0.8)

            ax2 = ax.twinx()
            ax2.grid(False)
            ax2.plot(years, avg_k, marker='o', markersize=3, linestyle='--', color=color_palette['line'], linewidth=1.5)

            ax2.spines['right'].set_color('grey')
            ax2.spines['right'].set_visible(True)
            ax2.set_ylabel('Capital Contribution (%)', color='black', labelpad=3)
            ax2.tick_params(axis='y', colors='black')

            if income_group in row1_groups:
                ax.set_ylim(row1_y_range)
            elif income_group == "Lower middle income":
                ax.set_ylim(lower_middle_y_range)
            elif income_group == "Low income":
                ax.set_ylim(low_income_y_range)
            else:
                max_burden = income_agg[income_agg['income_group'] == income_group][['log_capital', 'log_labor']].sum(axis=1).max()
                ax.set_ylim(0, max_burden * 1.05 if not np.isnan(max_burden) else 1)

            ax.set_xticks(selected_years)
            ax.set_xticklabels(selected_years, rotation=45, ha='right')
            ax.set_xlabel('Year', color=color_palette['text'], labelpad=5)
            ax.set_ylabel('Economic Burden (2024 U.S. $)', color=color_palette['text'], labelpad=10)

            ax2.set_ylim(0, 100)
            ax.set_title(income_group, fontweight='bold', color=color_palette['text'], pad=15)

            for i, (x, y) in enumerate(zip(years, avg_k)):
                if i in [0, len(years)-1]:
                    offset = 2.5 if y < 95 else -3.5
                    ax2.text(x, y + offset, f'{y:.3f}%', ha='center', color=color_palette['text'], fontsize=12)

        legend_elements = [
            plt.Rectangle((0,0),1,1, color=color_palette['capital'], label='Capital Burden'),
            plt.Rectangle((0,0),1,1, color=color_palette['labor'], label='Labor Burden'),
            plt.Line2D([0],[0], marker='o', color=color_palette['line'], label='Capital Contribution', linestyle='--', markersize=5)
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=3, frameon=False)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88, hspace=0.4, wspace=0.3)
        self.logger.info("Figure8: Trends in cumulative Economic Burden and Capital Contribution by income group")
        plt.savefig(os.path.join(self.output_dir, 'fig8.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)

    def run_full_process(self):
        self.fig1_visual()
        self.fig2_visual()
        self.fig3_visual()
        self.fig4_visual()
        self.fig5_visual()
        self.fig6_visual()
        self.fig7_visual()
        self.fig8_visual()
        self.fig9_visual()
        self.fig10_visual()
        self.fig11_visual()
        self.fig12_visual()
        print("数据处理完成")

