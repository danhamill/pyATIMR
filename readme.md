# pyATIMR

Repository for Antecedent Temperature Index Melt Rate analysis

A collection of scripts written by:
Daniel Hamill daniel.d.hamill@usace.army.mil

## dependencies
```
python 3.6.5
pandas - Data Wrangler
numpy - Numerical Operations
[Piecewise Linear Function Package](https://github.com/cjekel/piecewise_linear_fit_py) - Python Library for to fit piecewise linear functions
matplotlib - Plotting
pip - Python Package manager
openpyxl - Python Library for Writing to existing Excel Workbooks
xlrd - Python Library for Writing to existing Excel Workbooks
xlsxwriter - Python Library for writing  new Excel Workbooks
```

## Clone the repository to a working directory:

```
git clone https://github.com/danhamill/pyATIMR.git
```

If you dont have git for windows intalled, you can also download a [zip file](https://github.com/danhamill/pyATIMR/archive/master.zip) from the online repository.

## Installation

I recommend that this package be installed using the [Anaconda](https://www.anaconda.com/) python package manager.  It comes with a robust set pre-installed python packages for scientific computing.  It also allows for the creation of virtual environments that can be used to resolve package interdependencies.

Open an Anaconda a prompt in the directory where you cloned the repository and type:

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

Definition of flags:
```
-t path to excel workbook exported from dss with SnoTel temperature observations
-s path to excel workbook exported from dss with SnoTel SWE observations
-b base temperature in °F
-n path to n_segs text file explained above
-o name of output excel workbook
```
