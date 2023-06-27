import pandas as pd
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import paneltime as pt
import time
import tbf

window = 250

pt.options.pqdkm.set([0,0,0,2,2])
pt.options.supress_output.set(True)
pt.options.multicoll_threshold_max.set(30)


RETURNS = 'returns'
SIGNALS = 'ARIMA_prediction'
FILE = '20230621 - S&P500_2012_2023for_paneltime.csv'


def main():
  df = pd.read_csv(FILE, parse_dates=True, index_col='Date')
  df['signal'] = df[SIGNALS]**2
  back_test(df)


def get_var(df, signal, use_signal=True):
  pvar = [0.05, 0.01]
  z = norm.ppf(pvar)

  if use_signal:
    s = pt.execute(RETURNS,df, HF = 'signal' ,console_output=True)
    s.predict(signal)
    scoef = s.ll.args.args_d["omega"][1,0]
  else:
    s = pt.execute(RETURNS,df,console_output=True)
    s.predict()    
    scoef = 0
  sigma = s.ll.var_pred**0.5
  return - z[0]*sigma, - z[1]*sigma, sigma, s.ll.args.args_d['gamma'][0], s.ll.args.args_d['psi'][0], scoef, s.ll.LL, s.converged, s




def back_test(df):
  pt_95 = []
  pt_99 = []
  pt_sigma = []
  ret = []
  gammaarr = []
  llarr = []
  psiarr = []
  signals = []
  signalcoef = []

  t0=time.time()
  for i in range(len(df)-window-1):
    df_windows =pd.DataFrame(df.iloc[i:i+window])
    next_period_signal = df['signal'].iloc[i+window]
    rtrn = df[RETURNS].iloc[i+window]
    try:
      pt_95t, pt_99t, pt_sigmat, gamma, psi, signalcoeft, ll, conv, s =  get_var(df_windows,next_period_signal, use_signal=True)
    except np.linalg.LinAlgError:
      print('Singular matrix')
      pt_95t, pt_99t, ll, conv = 0,0,None, False
    print(f'VaR 95% {i}:{-pt_95t}, r:{rtrn}, vio:{rtrn<-pt_95t}, sgnl:{signalcoeft}, gamma:{gamma}, psi:{psi}, LL:{ll}, conv:{conv}')
    signals.append(next_period_signal)
    ret.append(rtrn)  
    gammaarr.append(gamma)
    llarr.append(ll)
    psiarr.append(psi)
    pt_95.append(pt_95t)
    pt_99.append(pt_99t)
    pt_sigma.append(pt_sigmat)
    signalcoef.append(signalcoeft)
    
  print(f'used {time.time()-t0}')
  
  n = len(pt_95)
  
  violations95 = (np.array(ret)<-np.array(pt_95))
  violations99 = (np.array(ret)<-np.array(pt_99))
  
  df_ret = plot_and_store(pt_99, pt_95, pt_sigma, ret, signalcoef, gammaarr, psiarr, ll, violations95, violations99, signals)
  
  
def plot_and_store(pt_99, pt_95, pt_sigma, ret, signalcoef, gammaarr, psiarr, ll, violations95, violations99, signals):
  df_res = pd.DataFrame()
  n = len(pt_99)
  for d,lbl,bar, plot in (
                (-np.array(pt_99), 'paneltime VaR 99%', False, False),
                (-np.array(pt_95), 'paneltime VaR 95%', False, True),
                (pt_sigma, 'paneltime standard deviation', False, True),
                (ret, 'normalized returns', False, True),
                (np.array(signalcoef)/100, 'signal coef', False, True),
                (gammaarr, 'gamma', False, True),
                (psiarr, 'psi', False, True),
                (ll, 'll', False, False),
                (violations95, 'violations', True, True), 
                (violations99, 'violations', True, False)
                ):
    if not plot:
      pass
    elif bar:
      plt.bar(np.arange(n), d, label=lbl, color = 'lightgrey')
    else:
      plt.plot(d, label=lbl)
    df_res[lbl] = d
  
  df_res.to_pickle('var_results.pd')
  print(f"For the 95% VaR there were {sum(violations95)/len(violations95)}% violations")
  print(f"95%: pof: {tbf.pof(violations95,0.95)[0]}, tbfi: {tbf.tbfi(violations95,0.95)[0]}, tbf {tbf.tbf(violations95,0.95)[0]}")
  


main()