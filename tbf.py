import numpy as np
import pandas as pd

def test_tbf_against_matematica():
  
  df = pd.read_csv("matematica_data.csv",delimiter=";", dtype=float)
  
  violations = np.array(df['returns'])<-np.array(df['normal95'])
  print(f"pof matematica: 0.46147(0.49694), pof python:{'%s (%s)' %pof(violations, 0.95)}")
  print(f"tbfi matematica: 88.491(0.0047475), tbfi python:{'%s (%s)' %tbfi(violations, 0.95)}")
  print(f"tbf matematica: 88.952(0.0055565), tbf python:{'%s (%s)' %tbf(violations, 0.95)}")

from scipy.stats import chi2

def tbf(violations,VaR):
  a, _ = pof(violations, VaR)
  b, _ = tbfi(violations, VaR)
  
  sign = 1-chi2.cdf(a+b, sum(violations) + 1) 
  return a + b, sign
  
def pof(violations, VaR):
  N = len(violations)
  x = sum(violations)
  p = 1 - VaR
  p_obs = x/N
  numr = np.log(((1-p)**(N-x))*(p**x))
  denom = np.log(((1-p_obs)**(N-x))*(p_obs**x))
  LR = -2*(numr - denom)
  sign = 1-chi2.cdf(LR, 1) 
  return LR, sign

def tbfi(violations, VaR):
  N = len(violations)
  r = np.arange(N)
  violations[0] = False
  dates = r[violations]
  n = np.append(dates[0],dates[1:] - dates[:-1])
  p = 1-VaR
  p_obs = 1/n
  numr = ((1-p)**(n-1))*p
  denom = ((1-p_obs)**(n-1))*p_obs
  LR = -2*np.log(numr/denom) 
  sign = 1-chi2.cdf(np.sum(LR), sum(violations)) 
  return np.sum(LR), sign