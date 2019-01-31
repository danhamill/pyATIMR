# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 13:14:48 2019

@author: RDCRLDDH
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pwlf
from openpyxl import load_workbook
import sys, getopt, ast, os
import warnings
warnings.filterwarnings("ignore")

# suppress divide and invalid warnings
np.seterr(divide='ignore')
np.seterr(invalid='ignore')

def formatter(tmp_data):
   tmp_data = tmp_data.drop('A', axis=1)
   names = tmp_data.loc[0].tolist()
   names[0] = 'Date'
   tmp_data.columns = names
   tmp_data = tmp_data.iloc[6:]
   tmp_data['Date'] = pd.to_datetime(tmp_data['Date'])
   tmp_data[names[1:]] = tmp_data[names[1:]].apply(lambda x: x.astype('float'))
   return tmp_data

def WY_generator(year):
   start = pd.datetime((year-1),9,1)
   end = pd.datetime(year, 8,30)
   return start, end

def TI_index(tmp_sub,base=32):
   tmp_sub = tmp_sub.apply(lambda x: x-base)
   return tmp_sub

def r2_calculator(my_pwlf):
   # calculate the piecewise R^2 value
   R2values = np.zeros(my_pwlf.n_segments)
   for i in range(my_pwlf.n_segments):
       # segregate the data based on break point locations
       xmin = my_pwlf.fit_breaks[i]
       xmax = my_pwlf.fit_breaks[i+1]
       xtemp = my_pwlf.x_data
       ytemp = my_pwlf.y_data
       indtemp = np.where(xtemp >= xmin)
       xtemp = my_pwlf.x_data[indtemp]
       ytemp = my_pwlf.y_data[indtemp]
       indtemp = np.where(xtemp <= xmax)
       xtemp = xtemp[indtemp]
       ytemp = ytemp[indtemp]
   
       # predict for the new data
       yhattemp = my_pwlf.predict(xtemp)
   
       # calcualte ssr
       e = yhattemp - ytemp
       ssr = np.dot(e, e)
   
       # calculate sst
       ybar = np.ones(ytemp.size) * np.mean(ytemp)
       ydiff = ytemp - ybar
       sst = np.dot(ydiff, ydiff)
   
       R2values[i] = 1.0 - (ssr/sst)
   return R2values

def df_merger(swe_sub, tmp_sub):
   swe_col,tmp_col = [],[]
   [swe_col.append(tuple([col,'SWE'])) for col in swe_sub.columns]
   swe_sub.columns=pd.MultiIndex.from_tuples(swe_col)
   [tmp_col.append(tuple([col,'TI'])) for col in tmp_sub.columns]
   tmp_sub.columns=pd.MultiIndex.from_tuples(tmp_col)
   merge  = pd.merge(swe_sub,tmp_sub, left_index=True, right_index=True, how='left')
   merge = merge.sort_index(level=0, axis=1)
   return merge


def regression_analysis(my_pwlf,col):
   r_sqd = r2_calculator(my_pwlf)
   breaks = my_pwlf.fit_breaks
   breaks = [ breaks[i:i+2] for i in range(len(breaks))][:-1]
   tmp = pd.concat([pd.DataFrame(breaks), pd.DataFrame(r_sqd),pd.DataFrame(my_pwlf.slopes),pd.DataFrame(my_pwlf.intercepts)],axis=1)
   tmp.columns = pd.MultiIndex.from_tuples([(col,'TI_Start'),(col,'TI_End'),(col,'r_sqd'),(col,'ATIMR'),(col,'Intercept')])
   return tmp

def gradient_merger(grad_df,col):
   grad_col = [(col,'ATI'), (col,'ATIMR')]
   grad_df.columns = pd.MultiIndex.from_tuples(grad_col)
   return grad_df
   
def mypercent(x):
   return np.percentile(x,75)

def composite_meltrate(grad_out):
   #calculate possible composite functions
   new_df = grad_out.stack(level=0)

   pct_75 = lambda y: np.percentile(y, 75)
   pct_75 .__name__ = 'pct_75 '
   
   pct_25 = lambda y: np.percentile(y, 25)
   pct_25 .__name__ = 'pct_25'
  
   #calculate possible composite functions
   func_list = [pct_25, np.median, np.mean ,pct_75]
   new_df = pd.pivot_table(new_df, values='ATIMR',index='ATI',aggfunc=func_list)
   new_df.columns = ['Composite_25_pct','Composite_Median','Composite_Mean','Composite_75_pct']
   new_df= new_df.drop_duplicates()
   return new_df


def export_analysis(stat_out, swe_sub, tmp_sub, comp_mr, out_book,year):
   '''
   Fuction to export the data to excel workbook
   Inputs: 
      stat_out = pandas dataframe contining the resluts of the regression analysis
      swe_sub  = pandas dataframe containing the SWE measurments for a WY
      tmp_sub =  pandas dataframe containing the TI calcualations for a WY
      comp_mr = pandas dataframe containing the composite TI vs. ATIMR results
      out_book = string representing the name of the outbook excel workbook
      year = string of the WY analyzed
   '''
   if os.path.exists(out_book):
      pass
   else:
      writer = pd.ExcelWriter(out_book,engine='xlsxwriter')
      writer.save()
      del writer
   stat_out = stat_out.stack(level=0).sort_index(level=1)
   stat_out = stat_out.drop(stat_out[stat_out.ATIMR >0].index)
   stat_out.ATIMR = stat_out.ATIMR * -1
   merge = df_merger(swe_sub.copy(), tmp_sub.copy())
   merge = merge.dropna(how='all',axis=0)
   writer = pd.ExcelWriter(out_book, engine='openpyxl')
   writer.book = load_workbook(out_book)
   merge.to_excel(writer, sheet_name=str(year),na_rep ='--')
   stat_out.to_excel(writer,sheet_name=str(year),startcol=16,na_rep ='--')
   comp_mr.to_excel(writer,sheet_name=str(year),startcol=25,na_rep = '--')
   writer.save()
   print('!!Saved analysis results to: ' + str(out_book))
    
    
def main(WY, tmp_data, swe_data):
   
   '''
   Main processing and plotting function
   Inputs:
      WY : STring of the water year being analyzed
      tmp_data: pandas data frame containing the temperature measurements exported from dss
      swe_dtat : pandas dataframe with the swe measurements exported from dss
   '''
   for year in [WY]:#2008, 2010, 2011
      print('!!Starting Analysis for WY ' + str(year))
      start, end = WY_generator(year)
      tmp_sub = tmp_data[(tmp_data['Date']>= start) & (tmp_data['Date']<= end)].set_index('Date')
      swe_sub = swe_data[(swe_data['Date']>= start) & (swe_data['Date']<= end)].set_index('Date')
      tmp_sub = tmp_sub.where(swe_sub.cummax() == swe_sub.max())
      swe_sub = swe_sub.where(swe_sub.cummax() == swe_sub.max())
      tmp_sub = TI_index(tmp_sub)
      tmp_sub[tmp_sub<0] = 0
      tmp_sub = tmp_sub.cumsum()
      
      #initialize dataframe for regression statistics
      stat_out = pd.DataFrame()
      
      #initilize dataframe for ATIMR Data
      grad_out = pd.DataFrame()
      
      
      num_segs = seg_dict[year]
      linestyles = ['-', '--', '-.', ':','-', '--', '-.']
      colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628']
      marks = ['+']
      n = 0
      fig,(ax,ax1,ax2) = plt.subplots(nrows=3,sharex=True)
      for col in swe_sub.columns:
         x_data = tmp_sub[[col]].dropna()
         y_data = swe_sub[[col]].dropna()
         
         #Remove zeros from swe measurements
         x_data = x_data[y_data>0]
         y_data = y_data[y_data>0].dropna()
         x_data = x_data.reindex(y_data.index)
         
         #set negative temp to zero
         x_data[x_data<0] = 0
   
         # do piecewise linear fititing
         my_pwlf  = pwlf.PiecewiseLinFit(x_data.values.flatten(), y_data.values.flatten())
         res = my_pwlf.fit(num_segs[n])
         
         slopes = my_pwlf.slopes
         
         reg_stats = regression_analysis(my_pwlf, col)
         stat_out = pd.concat([stat_out,reg_stats],axis=1) 
         
         #get lines for swe vs TI subplot
         xHat = np.linspace(min(x_data.values), max(x_data.values), num=10000)
         yHat = my_pwlf.predict(xHat)
         grad = np.where(np.gradient(yHat) >0)
         xHat[grad] = np.nan
         yHat[grad] = np.nan      
         
         #get data for ATIMR vs TI subplot
         xHat1 = np.arange(reg_stats[(col,'TI_Start')].min(),reg_stats[(col,'TI_End')].max(),1)
         yHat1 = np.ones(xHat1.shape)     
         for i in range(len(slopes)):
            yHat1[(xHat1 >= reg_stats[(col,'TI_Start')][i])& (xHat1 < reg_stats[(col,'TI_End')][i])] =slopes[i]          
         yHat1[yHat1>0] = np.nan
              
         grad_df = pd.DataFrame({'ATI':xHat1,'MR':yHat1*-1})
         grad_df = grad_df.ffill(axis=0).bfill()
         
         #do the plotting
         grad_df.plot(x='ATI',y='MR',ax=ax1,ls=linestyles[0],c=colors[n],label=col,sharex=ax1)
         points,= ax.plot(x_data, y_data, marker = marks[0], ls='None', c=colors[n])
         line, =ax.plot(xHat,yHat,linestyles[0],c=colors[n],label=col)
         
         grad_out = pd.concat([grad_out, gradient_merger(grad_df.copy(),col)],axis=1)
         n+=1
   
      comp_mr = composite_meltrate(grad_out)
      comp_mr.plot(ax=ax2)
      
      ax1.legend(loc='upper center', bbox_to_anchor=(0.5,1.25),ncol=4, fancybox=True, shadow=True,fontsize = 'x-small')
      ax2.legend(loc='upper center', bbox_to_anchor=(0.5,1.25),ncol=3, fancybox=True, shadow=True,fontsize = 'x-small')
      fig.suptitle('WY ' + str(year))
      ax1.set_ylim(0,0.15)
      ax2.set_ylim(0,0.15)
      ax1.set_xlabel('ATI[°F-Day]' )
      ax.set_ylabel('SWE [in]')
      ax1.set_ylabel('Melt Rate [in/°F-Day]')
      ax2.set_ylabel('Melt Rate [in/°F-Day]')
      fig.canvas.draw()
      fig.tight_layout()
      fig.subplots_adjust(top=0.933, hspace=0.25)
      fig.savefig(str(year) + '.png',dpi=600)
      print('!!Saved output plot to: ' + str(year) + '.png')
      

      export_analysis(stat_out, swe_sub, tmp_sub, comp_mr, out_book,year)


if __name__ =='__main__':
   
   #script, tmp_xlsx, swe_xlsx, base_temp, n_seg , out_book = sys.argv
   argv = sys.argv[1:]
   try:
      opts, args = getopt.getopt(argv,"h:t:s:b:n:o:")
   except getopt.GetoptError:
      print('Example usage: python pyATIMR\main.py -t data\Temp_Data.xlsx -s data\SWE_Data.xlsx -b 32 -n n_segs.txt -o output.xlsx')
      sys.exit(2)   
      
   for opt, arg in opts:
      if opt == ("-h"):
         print('Example usage: python pyATIMR\main.py -t data\Temp.xlsx -s data\Temp.xlsx -b 32 -n n_seg.txt -o output.xlsx')
         sys.exit()
      elif opt in ("-t"):
         tmp_data = arg
      elif opt in ("-s"):
         swe_data = arg  
      elif opt in ("-b"):
         base_temp = arg 
      elif opt in ("-n"):
         n_seg = arg 
      elif opt in ("-o"):
         out_book = arg

          
   #paths to exported data from DSS file   
   tmp_data = pd.read_excel(tmp_data)
   swe_data = pd.read_excel(swe_data)
   
   base_temp = float(base_temp)
   
   with open(n_seg) as f: #'labels.txt') as f:
      cols = f.readlines()
      
      
   WYs = [int(x.split(':')[0]) for x in cols]
   seg_data = [x.split(':')[1].strip() for x in cols]
   seg_data = [ast.literal_eval(x) for x in seg_data]
   seg_dict = dict(zip(WYs,seg_data))
   
   
   #Format input xlsx files
   tmp_data, swe_data = [formatter(i) for i in [tmp_data,swe_data]]
   

   seg_dict = {2008:[3,5,2,5,5,5,5],
               2010:[3,6,6,4,4,4,6],
               2011:[3,3,2,3,3,3,3]}
   
   #name of outbook excel workbook
   out_book = out_book
   
   #iterate thrugh all water years supplied in seg dict
   for year in seg_dict.keys():
      main(year, tmp_data, swe_data)
   
