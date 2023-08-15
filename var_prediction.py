import pandas as pd
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import paneltime as pt
import time
import tbf
import os
import zipfile


pt.options.supress_output.set(True)
pt.options.multicoll_threshold_max.set(30)
pt.options.max_iterations.set(150)


PVAR = [0.05, 0.01]
RETURNS = 'returns'
PRED_RETURNS = 'returns predicted'
SIGNAL = 'signal'
FORCE_RUN = True
ZIPFILE = 'data.zip'
datadir = ZIPFILE.replace('.zip','')
OUTPUT_DEST ='output'
INPUT_SRC='data'

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)  

def main():
  if not os.path.exists(datadir):
    os.makedirs(datadir)
    
  if not os.path.isfile(datadir + '/' + 'sp500_index.csv'):
    zipfile.ZipFile(ZIPFILE, 'r').extractall(datadir)
    
  if not os.path.exists(OUTPUT_DEST):
    os.makedirs(OUTPUT_DEST)
       
  #defines the names of the price variables
  pricenames = {'sp500_index.csv':'S&P500'}
  
  df_stat = pd.DataFrame(index=pd.Index([], name='Method'))
  for f in os.listdir(datadir): 
    for prediction in [True, False]:
      if f in pricenames:
        d, fname = back_test_data(f, pricenames[f], 250, prediction)
        df_stat = pd.concat((df_stat, d))
  print(df_stat)
  df_stat.to_pickle(OUTPUT_DEST + '/' + fname+'_allstats.pd')

def back_test_data(filename, pricename, window, predict):
  
  #defining file names
  s = ''
  if predict:
    s = '_pred'
  fnameroot = filename.replace('.csv','')
  input_f_name = INPUT_SRC + '/'+ filename
  df_file_name =  OUTPUT_DEST + '/' + fnameroot + '.pd'
  backtest_file_name =  OUTPUT_DEST + '/' + fnameroot + s + '_backtest.pd'
  stats_file_name =  OUTPUT_DEST + '/' + fnameroot + s + '_stats.pd'
  
  
  #if data has been saved in a previous session, then load it, if not, open the data, calcuate returns and save it:
  if os.path.isfile(df_file_name):
    df = pd.read_pickle(df_file_name)
  else:
    df = pd.read_csv(input_f_name, parse_dates=True, index_col='Date')
    df['returns']=np.log(df[pricename]).diff() 
    df.to_pickle(df_file_name)
   
  #predicting returns if not all ready done, and creating a signal variable (only for paneltime)
  df = df.dropna()
  if not PRED_RETURNS in df:
    estimate_returns(df, window)
    df[SIGNAL] = 1000*df[PRED_RETURNS]**2
  print(f"Correlation between predicted and actual returns: {df[RETURNS].corr(df[PRED_RETURNS])}")
  df.to_pickle(df_file_name)
  
  df = df.dropna()
  #back testing if not all ready done:
  if os.path.isfile(backtest_file_name):
    df_res = pd.read_pickle(backtest_file_name)
  else:
    df_res = back_test(df, window, predict)
    df_res.to_pickle(backtest_file_name)
    
  
  #Calculating test statistics:
  df_stats_table = calc_statistics(df_res, predict, fnameroot)
  df_stats_table.to_pickle(stats_file_name)
  

  print(df_stats_table)
  
  return df_stats_table, fnameroot

def extract_rar(path):
  if not os.path.exists('data'):
    os.makedirs('d')
    
  # Specify the path to the RAR archive
  rar_path = 'path/to/archive.rar'
  
  # Specify the path to the CSV file within the RAR archive
  file_path_within_rar = 'path/within/archive/file.csv'
  
  # Open the RAR archive
  with rarfile.RarFile(path) as rf:
      # Extract the CSV file to a temporary location
      rf.extract(file_path_within_rar, path='temp')
  
  # Read the CSV file using pandas
  df = pd.read_csv('temp/' + file_path_within_rar)
  
  # Perform operations with the DataFrame as needed
  
  # Finally, delete the temporary extracted file
  os.remove('temp/' + file_path_within_rar)
  
  

def estimate_returns(df, window):
  df[PRED_RETURNS] = np.nan
  print('predicting ...')
  pt.options.pqdkm.set([2,2,0,2,2])
  pt.options.tolerance.set(0.001)
  N=len(df)-window-1
  for i in range(N):
    ret = df[RETURNS].iloc[i+window]
    df_windows = pd.DataFrame(df.iloc[i:i+window])
    s = pt.execute(RETURNS,df_windows,console_output=True)
    s.predict()
    print(f"iter {i}; actual return: {ret}, estimated: {s.ll.u_pred}")
    
    df.loc[df.index[i+window],PRED_RETURNS] = s.ll.u_pred
    
    
def paneltime_est(df, signal, prev_sigma, predict):
  z = norm.ppf(PVAR)
  s = pt.execute(RETURNS,df, HF = SIGNAL , console_output=True)
  if predict:
    s.predict(signal)
  else:
    s.predict()
  sigma = s.ll.var_pred**0.5
  return - z[0]*sigma, - z[1]*sigma, sigma

def normal_est(df, signal, predict):
  #using signal
  if predict:
    x = np.append(df[RETURNS], signal)
  else:
    x = df[RETURNS]
  z = norm.ppf(PVAR)
  sigma = np.std(x, ddof=1)
  return - z[0]*sigma, - z[1]*sigma, sigma

def historcal_est(df, signal, predict):
  #using signal
  if predict:
    x = np.append(df[RETURNS], signal)
  else:
    x = df[RETURNS]
  return abs(np.quantile(x, PVAR[0])), abs(np.quantile(x, PVAR[1]))

s2tmp=[]
rettmp=[]

def ewma_est(df, s, initvar, signal, predict):
  #using signal
  if predict:
    x = np.append(df[RETURNS], signal)
  else:
    x = df[RETURNS]
  z = norm.ppf(PVAR)
  lmbda = 0.94
  r_prev = x[-1]
  if len(s)==0:
    sigma = initvar
  else:
    sigma = s.iloc[-1]
  sigma2 = (1-lmbda)*r_prev**2 + lmbda*sigma**2
  s2tmp.append(sigma2)
  rettmp.append(r_prev)
  sigma = sigma2**0.5
  return - z[0]*sigma, - z[1]*sigma, sigma



  
def back_test(df, window, predict):
  pt.options.pqdkm.set([0,0,0,2,2])
  pt.options.tolerance.set(0.001)
  df_res = pd.DataFrame(columns=['EWMA_sigma'])
  
  #defining a dictionary for adding the items for each iteration to the data frame
  #EWMA_sigma and Paneltime_sigma are inputs, so needs to be defined first
  d = {'Paneltime_sigma':None}

  N = len(df)-window-1
  for i in range(N):
    #defining som variables
    df_windows =pd.DataFrame(df.iloc[i:i+window])
    d['signal'] = df[SIGNAL].iloc[i+window]
    d['pred_returns'] = df[PRED_RETURNS].iloc[i+window]
    d['return'] = df[RETURNS].iloc[i+window]
    d['Date'] = df.index[i+window]
    #obtaining the different VaR estimates
    d['Paneltime95'], d['Paneltime99'], d['Paneltime_sigma']   =  paneltime_est(df_windows, d['signal'], d['Paneltime_sigma'] , predict)
    d['Normal95'], d['Normal99'], d['Normal_sigma']            =  normal_est(df_windows,d['pred_returns'], predict)
    d['Historical95'], d['Historical99']                       =  historcal_est(df_windows,d['pred_returns'], predict)
    d['EWMA95'], d['EWMA99'], d['EWMA_sigma']                  =  ewma_est(df_windows, df_res['EWMA_sigma'], d['Normal_sigma'], d['pred_returns'] , predict)

    
    print(f"{i}, 95% VaRs pt:{d['Paneltime95']}, norm:{d['Normal95']}, hist:{d['Historical95']}, EWMA:{d['EWMA95']} | "
          f"95% violations pt:{-d['Paneltime95']>d['return']}, norm:{-d['Normal95']>d['return']}, hist:{-d['Historical95']>d['return']}, EWMA:{-d['EWMA95']>d['return']}")
    
    #adding it to the df
    df_d = pd.DataFrame(d, index=[0])
    df_res = pd.concat((df_res, df_d), ignore_index=True)
    
    
  return df_res
  
  
def calc_statistics(df, predict, fname):
  df_stat = pd.DataFrame(index=pd.Index([], name='Method'))
  for p in ['95','99']:
    for method in ['Paneltime', 'Normal', 'Historical', 'EWMA']:
      name = method+p
      d = tbf.tbf(-df[method+p]>df['return'], float(p)/100)
      d['File'] = fname
      d['Prediction'] = predict
      df_d = pd.DataFrame(d, index=pd.Index([name], name='Method'))
      df_stat = pd.concat((df_stat, df_d))

  return df_stat





main()