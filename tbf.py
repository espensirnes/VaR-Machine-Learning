import numpy as np
import pandas as pd

def test_tbf_against_matematica():
  
  df = pd.read_csv("data/matematica_data.csv",delimiter=";", dtype=float)
  
  violations = np.array(df['returns'])<-np.array(df['normal95'])
  print(f"pof matematica: 0.46147(0.49694), pof python:{'%s (%s)' %pof(violations, 0.95)}")
  print(f"tbfi matematica: 88.491(0.0047475), tbfi python:{'%s (%s)' %tbfi(violations, 0.95)}")
  print(f"tbf matematica: 88.952(0.0055565), tbf python:{'%s (%s)' %tbf(violations, 0.95)}")

from scipy.stats import chi2

def tbf(violations,VaR):
  d = {}
  d['LRatioPOF'], d['PValuePOF'] = pof(violations, VaR)
  
  (d['LRatioTBFI'], d['PValueTBFI'] , 
   d['TBFMin'],    d['TBFQ1'],    d['TBFQ2'],   d['TBFQ3'],    d['TBFMax']) = tbfi(violations, VaR)
  
  d['N'] = len(violations)
  if d['LRatioTBFI'] is None:
    d['LRatioTBF'] = None
    d['PValueTBF'] = None
    return 
  
  d['LRatioTBF'] = d['LRatioTBFI'] + d['LRatioPOF']
  
  d['PValueTBF'] = 1-chi2.cdf(d['LRatioTBF'] , sum(violations) + 1) 
  
  return d
  
def pof(violations, VaR):
  N = len(violations)
  x = sum(violations)
  if x ==0:
    return None, None
  p = 1 - VaR
  p_obs = x/N
  numr = (N-x)*np.log(1-p) + x* np.log(p)
  denom = (N-x)*np.log(1-p_obs) + x*np.log(p_obs)
  LR = -2*(numr - denom)
  sign = 1-chi2.cdf(LR, 1) 
  return LR, sign

def tbfi(violations, VaR):
  if sum(violations)==0:
    return [None]*7
  N = len(violations)
  r = np.arange(N)
  violations[0] = False
  dates = r[violations]
  n = np.append(dates[0],dates[1:] - dates[:-1])
  p = 1-VaR
  p_obs = 1/n
  numr = (n-1)*np.log(1-p) + np.log(p)
  denom = (n-1)*np.log(1-p_obs + (n==1)) + np.log(p_obs)
  LR = -2*(numr - denom) 

  sign = 1-chi2.cdf(np.sum(LR), sum(violations)) 
  return [np.sum(LR), sign] + quartiles(n)

def quartiles(n):
  q1 = np.percentile(n, 25)
  q2 = np.percentile(n, 50)
  q3 = np.percentile(n, 75)
  tbf_max = np.max(n)
  tbf_min = np.min(n)
  return [tbf_min, q1, q2, q3, tbf_max]