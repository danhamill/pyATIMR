# pyATIMR

Repository for Antecedent Temperature Index Melt Rate analysis

A collection of scripts written by:
Daniel Hamill daniel.d.hamill@usace.army.mil

## dependencies
```
pandas - Data Wrangler
numpy - Numerical Operations
[Piecewise Linear Function (pwlf)](https://github.com/cjekel/piecewise_linear_fit_py) - Python Library for to fit piecewise linear functions
matplotlib - Plotting
pip - Python Package manager
openpyxl = Python Library for Writing Excel Workbooks
```


Installation:
It recommended python package be installed using [Anaconda](https://www.anaconda.com/) python package manager.  From anaconda prompt:

```
conda env create -f tf_env.yml
```

After the anaconda environment `pyATIMR` is created, you can enter it with:

```
activate pyATIMR
```


## Relevant Files

`n_segs.txt` - Text file containing the water years and number of liner segments to fit to a sequence of snotel sites
```
Format
water_year1:[ num linear segments to fit to Snotel Site 1 , num linear segments to fit to Snotel Site 2 ...],
water_year2:[ num linear segments to fit to Snotel Site 1 , num linear segments to fit to Snotel Site 2 ...],

Example:
2008: [3,5,2,5,5,5,5]
```

`data\Temp_Data.xlsx` - Excel workbook containing SnoTel Temperature observations exported from DSS

`data\SWE_Data.xlsx` - Excel workbook containing SnoTel SWE observations exported from DSS

`py_env.yml` = Configuration file for anaconda environment

## Running the analysis

```
python pyATIMR\main.py -t data\Temp_Data.xlsx -s data\SWE_Data.xlsx -b 32 -n n_segs.txt -o output.xlsx
```
