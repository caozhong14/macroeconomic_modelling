# Macroeconomic Modelling Project

This project is designed for macroeconomic modelling and empirical analysis. It provides a complete workflow including data processing, model construction, and result generation.

## Directory Structure

```text
.
├── main.py              # Main entry script to run the entire pipeline
├── config_yml/              # Configuration files (to be prepared by the user)
├── database/            # Data directory (not tracked by Git)
│   ├── bigdata/         # Raw or large-scale datasets
│   └── data/            # Processed or intermediate datasets
├── models/              # Core source code (models, functions, utilities)
├── application/         # visualization, postprocess
├── output/              # Output results (figures, tables, logs, etc.)
├── README.md            # Project documentation
└── .gitignore           # Git ignore rules
````

## How to Run

After setting up the environment (requirement.txt) and preparing the required data, run the entire project with:

```bash
python main.py
```

The `main.py` script serves as the unified entry point and will sequentially execute all steps of the analysis.

## Configuration Files

* Configuration files in the `config_yml/` directory **must be prepared by the user**
* These files typically define paths, parameters, and model settings
* Please adjust the configuration according to your local environment and research needs

## Data Availability

* Data files under the `database/` directory are **not included in this repository**
* To obtain the required datasets, please **contact the author**
* Once obtained, place the data in the corresponding `database/bigdata/` or `database/data/` directories

## Contact

For questions, data access, or collaboration, please contact **Zhong Cao**.

```
Zhong Cao
Heidelberg University
zhong.cao@uni-heidelberg.de
```
