#read FHSI_minute_20051201to20190326.hdf5
import os
import numpy as np
from pandas import HDFStore,DataFrame
import pandas as pd
from datetime import datetime as dt
import datetime

folder_path=r"C:\Users\alexlau\Dropbox\notebooks\index_analysis\mis"
fn = os.path.join(folder_path,"FHSI_minute_20051201to20190326.hdf5")
store = pd.HDFStore(fn)
print(store)
data_all_final= store.select('FHSI_minute')
 
store.close()


data_all_final_check=data_all_final.head(10)
data=data_all_final[['date_ymd_hms','date_ymd','Open','High','Low','Close']].copy()
data=data.rename(columns={'date_ymd_hms':'Date1','date_ymd':'Date2'})

#read prediction
main_dir=r'C:\Users\alexlau\Dropbox\notebooks\index_analysis'

from pandas import read_excel
from datetime import datetime as dt
train_test_Setting = read_excel(r'C:\Users\alexlau\Dropbox\notebooks\index_analysis\index_table_v2.xlsx','Sheet2')
train_test_Setting=train_test_Setting.loc[train_test_Setting['run_yes']=='yes',:]
train_test_Setting=train_test_Setting.reset_index(drop=True)

d0=pd.DataFrame([])
all_number=train_test_Setting['Number'].values.tolist()
i=0
for i in range(0,train_test_Setting.shape[0]):
    file_name=str(train_test_Setting['Number'][i])+'_test_'+str(train_test_Setting['Test_start'][i].strftime("%Y"))+'.xlsx'
    d1 = read_excel(os.path.join(main_dir,'plot',file_name),'daily_detail_summary')
    d0=d0.append(d1)
    print('finished ',i,' out of ',train_test_Setting.shape[0])


d0.to_csv(os.path.join(folder_path,"all_prediction.csv"))










d0=pd.read_csv(os.path.join(folder_path,"all_prediction.csv"))


original_guess=d0[['Date2','Y_up_predict']].copy()
original_guess=original_guess.rename(columns={'Y_up_predict':'prediction'})
original_guess=original_guess.reset_index(drop=True)


all_in_data=data.Date2.unique().tolist()


original_guess['have_data']=original_guess['Date2'].apply(lambda x: x in all_in_data)

original_guess=original_guess.loc[original_guess['have_data']==True,:]

original_guess=original_guess.loc[original_guess['Date2']<='2013-12-31',:]

original_guess=original_guess.reset_index(drop=True)






from random import *
 
from datetime import datetime as dt
import datetime
from collections import OrderedDict
import pandas as pd
import numpy as np
 

morning_signal=0
profit_target=20000
stop_level=200
after_stop_target=20000
after_stop_stop=20000
date_use="2011-01-04"
date_ymd_col_name='Date2'
date_ymd_hms_col_name='Date1'
second_stage_trade=True
start_row=0 
end_row=-1 

 
start_row=0 #9:15 Open
start_row=5 #9:20 Open
end_row=15 #0929 close
end_row=-1 # 1629 close
end_row=-2 # 1628 close


 
def strategy1(data,date_use,date_ymd_col_name,date_ymd_hms_col_name,morning_signal,profit_target,stop_level,after_stop_target,after_stop_stop,second_stage_trade,start_row=0,end_row=-1):
    data_use=data.loc[data[date_ymd_col_name]==date_use,:].copy()
    data_use=data_use.reset_index(drop=True)
    if end_row>=0:
        data_use=data_use[(start_row):(end_row)]
    else:
        data_use=data_use[(start_row):(data_use.shape[0]+1+end_row)]
 
    data_use=data_use.reset_index(drop=True)
 
    entry_price=data_use[0:1]['Open'][0]
    exit_price=data_use[0:1]['Open'][0]
    second_entry_price=999999
    second_exit_price=999999
    trigger_first_stop=False
    trigger_second_stop=False
#    date_use_dt=dt.strptime(date_use,"%Y-%m-%d")
#    exit_time=dt(int(date_use_dt.strftime("%Y")),int(date_use_dt.strftime("%m")),int(date_use_dt.strftime("%d")),16,30,0)
    achieve_profit=999999
    achieve_stop=999999
    achieve_profit2=999999
    achieve_stop2=999999    
 
    if morning_signal==1:
        #find when achieve profit target
        data_use.loc[data_use['High']>=entry_price+profit_target,'indicate_profit']=1
        data_use['indicate_profit']=data_use['indicate_profit'].fillna(0)
        if sum(data_use['indicate_profit']==1)>0:
            achieve_profit=data_use.index[data_use['indicate_profit']==1][0]
        else:
            achieve_profit=999999
 
        #find when achieve stop target        
        data_use.loc[data_use['Low']<=entry_price-stop_level,'indicate_stop']=1
        data_use['indicate_stop']=data_use['indicate_stop'].fillna(0)
        if sum(data_use['indicate_stop']==1)>0:
            achieve_stop=data_use.index[data_use['indicate_stop']==1][0]
        else:
            achieve_stop=999999
 
        #case1: achieve_profit=999999, achieve_stop=999999, leave at close
        if (achieve_profit==999999)&(achieve_stop==999999):
            exit_price=data_use.loc[data_use['Date1']==max(data_use['Date1']),'Close'].values[0]
 
 
        #case2 or case 5, second trade
        if ((achieve_profit==999999)&(achieve_stop!=999999))|((achieve_profit!=999999)&(achieve_stop!=999999)&(achieve_profit>achieve_stop)):
            trigger_first_stop=True
            exit_price=entry_price-stop_level
 
        #case3 or case 4, exit at profit target
        if ((achieve_profit!=999999)&(achieve_stop==999999))|((achieve_profit!=999999)&(achieve_stop!=999999)&(achieve_profit<achieve_stop)):
            exit_price=entry_price+profit_target
 
        #case6, second trade
        if (achieve_profit!=999999)&(achieve_stop!=999999)&(achieve_profit==achieve_stop):
            trigger_first_stop=True
            exit_price=entry_price-stop_level
 
        if (trigger_first_stop==True)&(second_stage_trade==True):
            data_use2=data_use[achieve_stop:].copy()  
            data_use2=data_use2.reset_index(drop=True)
            second_entry_price=entry_price-stop_level-1
 
            #find when achieve profit target
            data_use2.loc[data_use2['Low']<=second_entry_price-after_stop_target,'indicate_profit2']=1
            data_use2['indicate_profit2']=data_use2['indicate_profit2'].fillna(0)
            if sum(data_use2['indicate_profit2']==1)>0:
                achieve_profit2=data_use2.index[data_use2['indicate_profit2']==1][0]
            else:
                achieve_profit2=999999
 
            #find when achieve stop target        
            data_use2.loc[data_use2['High']>=(second_entry_price+after_stop_stop),'indicate_stop2']=1
            data_use2['indicate_stop2']=data_use2['indicate_stop2'].fillna(0)
            if sum(data_use2['indicate_stop2']==1)>0:
                achieve_stop2=data_use2.index[data_use2['indicate_stop2']==1][0]
            else:
                achieve_stop2=999999
 
            #case1: achieve_profit=999999, achieve_stop=999999, leave at close
            if (achieve_profit2==999999)&(achieve_stop2==999999):
                second_exit_price=data_use2.loc[data_use2['Date1']==max(data_use2['Date1']),'Close'].values[0]
 
            #case2 or case 5, second trade
            if ((achieve_profit2==999999)&(achieve_stop2!=999999))|((achieve_profit2!=999999)&(achieve_stop2!=999999)&(achieve_profit2>achieve_stop2)):
                trigger_second_stop=True
                second_exit_price=second_entry_price+after_stop_stop
 
            #case3 or case 4, exit at profit target
            if ((achieve_profit2!=999999)&(achieve_stop2==999999))|((achieve_profit2!=999999)&(achieve_stop2!=999999)&(achieve_profit2<achieve_stop2)):
                second_exit_price=second_entry_price-after_stop_target
 
            #case6, achieve_profit2==achieve_stop2, second trade
            if (achieve_profit2!=999999)&(achieve_stop2!=999999)&(achieve_profit2==achieve_stop2):
                trigger_second_stop=True
                second_exit_price=second_entry_price+after_stop_stop
 
    if morning_signal==0:
        #find when achieve profit target
        data_use.loc[data_use['Low']<=entry_price-profit_target,'indicate_profit']=1
        data_use['indicate_profit']=data_use['indicate_profit'].fillna(0)
        if sum(data_use['indicate_profit']==1)>0:
            achieve_profit=data_use.index[data_use['indicate_profit']==1][0]
        else:
            achieve_profit=999999
 
        #find when achieve stop target        
        data_use.loc[data_use['High']>=entry_price+stop_level,'indicate_stop']=1
        data_use['indicate_stop']=data_use['indicate_stop'].fillna(0)
        if sum(data_use['indicate_stop']==1)>0:
            achieve_stop=data_use.index[data_use['indicate_stop']==1][0]
        else:
            achieve_stop=999999
 
        #case1: achieve_profit=999999, achieve_stop=999999, leave at close
        if (achieve_profit==999999)&(achieve_stop==999999):
            exit_price=data_use.loc[data_use['Date1']==max(data_use['Date1']),'Close'].values[0]
 
        #case2 or case 5, second trade
        if ((achieve_profit==999999)&(achieve_stop!=999999))|((achieve_profit!=999999)&(achieve_stop!=999999)&(achieve_profit>achieve_stop)):
            trigger_first_stop=True
            exit_price=entry_price+stop_level
 
        #case3 or case 4, exit at profit target
        if ((achieve_profit!=999999)&(achieve_stop==999999))|((achieve_profit!=999999)&(achieve_stop!=999999)&(achieve_profit<achieve_stop)):
            exit_price=entry_price-profit_target
 
        #case6, second trade
        if (achieve_profit!=999999)&(achieve_stop!=999999)&(achieve_profit==achieve_stop):
            trigger_first_stop=True
            exit_price=entry_price+stop_level
 
        if (trigger_first_stop==True)&(second_stage_trade==True):
            data_use2=data_use[achieve_stop:].copy()
            data_use2=data_use2.reset_index(drop=True)
            second_entry_price=entry_price+stop_level+1
 
            #find when achieve profit target
            data_use2.loc[data_use2['High']>=second_entry_price+after_stop_target,'indicate_profit2']=1
            data_use2['indicate_profit2']=data_use2['indicate_profit2'].fillna(0)
            if sum(data_use2['indicate_profit2']==1)>0:
                achieve_profit2=data_use2.index[data_use2['indicate_profit2']==1][0]
            else:
                achieve_profit2=999999
 
            #find when achieve stop target        
            data_use2.loc[data_use2['Low']<=(second_entry_price-after_stop_stop),'indicate_stop2']=1
            data_use2['indicate_stop2']=data_use2['indicate_stop2'].fillna(0)
            if sum(data_use2['indicate_stop2']==1)>0:
                achieve_stop2=data_use2.index[data_use2['indicate_stop2']==1][0]
            else:
                achieve_stop2=999999
 
            #case1: achieve_profit=999999, achieve_stop=999999, leave at close
            if (achieve_profit2==999999)&(achieve_stop2==999999):
                second_exit_price=data_use2.loc[data_use2['Date1']==max(data_use2['Date1']),'Close'].values[0]
 
 
            #case2 or case 5, second trade
            if ((achieve_profit2==999999)&(achieve_stop2!=999999))|((achieve_profit2!=999999)&(achieve_stop2!=999999)&(achieve_profit2>achieve_stop2)):
                trigger_second_stop=True
                second_exit_price=second_entry_price-after_stop_stop
 
            #case3 or case 4, exit at profit target
            if ((achieve_profit2!=999999)&(achieve_stop2==999999))|((achieve_profit2!=999999)&(achieve_stop2!=999999)&(achieve_profit2<achieve_stop2)):
                second_exit_price=second_entry_price+after_stop_target
 
            #case6, achieve_profit2==achieve_stop2, second trade
            if (achieve_profit2!=999999)&(achieve_stop2!=999999)&(achieve_profit2==achieve_stop2):
                trigger_second_stop=True
                second_exit_price=second_entry_price-after_stop_stop
 
    output=pd.Series([date_use,morning_signal,entry_price,exit_price,trigger_first_stop,second_entry_price,second_exit_price,trigger_second_stop,
                      profit_target,stop_level,after_stop_target,after_stop_stop,second_stage_trade,achieve_profit,achieve_stop,achieve_profit2,achieve_stop2,start_row,end_row])
    return output
 
 
 
 
 
import time
import os
#parameter_df=pd.DataFrame(OrderedDict({'profit_target':     [20000,100,150,200,250,300,350,150,200,250,300,350,200,250,300,350],
#                                       'stop_level'   :     [200,75,75,75,75,75,75,100,100,100,100,100,150,150,150,150],
#                                       'after_stop_target': [10000,75,75,75,75,75,75,100,100,100,100,100,150,150,150,150],
#                                       'after_stop_stop':   [10000,75,75,75,75,75,75,100,100,100,100,100,150,150,150,150],
#                                       'second_stage_trade':[False,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True]}))

parameter_df = read_excel(r'C:\Users\alexlau\Dropbox\notebooks\index_analysis\mis\parameter_df.xlsx','Sheet1')
    
    
    
    
    
    
    
summary=pd.DataFrame()
 
 
start_time=dt.now()
i=30
j=0
for j in range(0,parameter_df.shape[0]):
    parameter_df_use=parameter_df[j:j+1]
    pt1=parameter_df_use['profit_target'].values[0]
    st1=parameter_df_use['stop_level'].values[0]
    pt2=parameter_df_use['after_stop_target'].values[0]
    st2=parameter_df_use['after_stop_stop'].values[0]
    second_stage_trade=parameter_df_use['second_stage_trade'].values[0]
    start_row=parameter_df_use['start_row'].values[0]
    end_row=parameter_df_use['end_row'].values[0]
 
    store_result=pd.DataFrame(columns=["Date","Prediction","entry_price","exit_price","trigger_first_stop",
                                       "second_entry_price","second_exit_price","trigger_second_stop",
                                       "profit_target","stop_level","after_stop_target","after_stop_stop","second_stage_trade",'achieve_profit','achieve_stop','achieve_profit2','achieve_stop2','start_row','end_row'])    
    for i in range(0,original_guess.shape[0]):
        row_use=original_guess[i:i+1]
        temp=strategy1(data,row_use['Date2'].values[0],'Date2','Date1',row_use['prediction'].values[0],pt1,st1,pt2,st2,second_stage_trade,start_row,end_row)
        temp=temp.values.reshape(1,temp.shape[0])
        temp=pd.DataFrame(temp)
        temp.columns=("Date","Prediction","entry_price","exit_price","trigger_first_stop","second_entry_price","second_exit_price",
                        "trigger_second_stop","profit_target","stop_level","after_stop_target","after_stop_stop","second_stage_trade",'achieve_profit','achieve_stop','achieve_profit2','achieve_stop2','start_row','end_row')
        store_result=store_result.append(temp)
        print("finished ",row_use['Date2'].values[0])
 
 
    store_result['year']=store_result['Date'].str[0:4]    
    store_result['first_commission']=12
    store_result.loc[store_result['second_entry_price']!=999999,'second_commission']=12
    store_result.loc[store_result['second_entry_price']==999999,'second_commission']=0
    store_result.loc[store_result['Prediction']==0,'Prediction']=-1
    store_result['Prediction_second']=store_result['Prediction']*-1
    store_result['pnl_first_trade']=(store_result['exit_price']-store_result['entry_price'])*store_result['Prediction']*10-store_result['first_commission']
    store_result['pnl_second_trade']=(store_result['second_exit_price']-store_result['second_entry_price'])*store_result['Prediction_second']*10-store_result['second_commission']
    store_result['pnl']=store_result['pnl_first_trade']+store_result['pnl_second_trade']
 
    file_name="hsi_investigate2_"+str(pt1)+"_"+str(st1)+"_"+str(pt2)+"_"+str(st2)+"_"+str(start_row)+"_"+str(end_row)+'_'+time.strftime("%Y%m%d")+'_'+time.strftime("%H%M%S")+".csv"
    save_path=os.path.join(main_dir,'plot2',file_name)
    store_result.to_csv(save_path,index=False)
 
    def pnl_function(x):
        year_name=x['year'].values[0]
        x['Cum_pnl_peryear']=x['pnl'].cumsum()
        MDD=max(np.maximum.accumulate(x['Cum_pnl_peryear']) - x['Cum_pnl_peryear'])
        max_downside=min(x['Cum_pnl_peryear'])
        final_cum_pnl=x['Cum_pnl_peryear'].values[-1]
        accuracy=sum(x['pnl']>0)/x.shape[0]
        return pd.Series([year_name,final_cum_pnl,MDD,max_downside,accuracy])
 
    temp=store_result.groupby(["year"]).apply(lambda x:pnl_function(x.reset_index(drop=True)))
    temp.columns = ('year',"FinalCumPnl","MDD","MaxDownside","accuracy")
 
    temp['profit_target']=pt1
    temp['stop_level']=st1
    temp['after_stop_target']=pt2
    temp['after_stop_stop']=st2
    temp['second_stage_trade']=second_stage_trade
    temp['start_row']=start_row
    temp['end_row']=end_row
 
    summary=summary.append(temp)
 
save_path=os.path.join(main_dir,'plot2','summary_pnl'+'_'+time.strftime("%Y%m%d")+'_'+time.strftime("%H%M%S")+'.csv')
summary.to_csv(save_path,index=False)
 
 
end_time=dt.now()
total_time=(end_time-start_time).total_seconds()
total_time
 
 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates, ticker
import matplotlib as mpl
import sys
sys.path.append(r'C:\Users\alexlau\Desktop\python\mpl_finance\dist\mpl_finance-0.10.0')
from mpl_finance import candlestick_ohlc
 
mpl.style.use('default')
 
 
data['Date1_string']=data['Date1'].apply(lambda x:x.strftime("%Y-%m-%d %H:%M:%S"))
data_for_plot=data.loc[data['Date2']=='2011-01-04',['Date1_string','Open','High','Low','Close']].copy()
data_for_plot['Open']=data_for_plot['Open'].astype(str)
data_for_plot['High']=data_for_plot['High'].astype(str)
data_for_plot['Low']=data_for_plot['Low'].astype(str)
data_for_plot['Close']=data_for_plot['Close'].astype(str)
data_for_plot=[tuple(r) for r in data_for_plot.values]
 
ohlc_data = []
 
for line in data_for_plot:
    ohlc_data.append((dates.datestr2num(line[0]), np.float64(line[1]), np.float64(line[2]), np.float64(line[3]), np.float64(line[4])))
 
fig, ax1 = plt.subplots()
candlestick_ohlc(ax1, ohlc_data, width = 0.5/(24*60), colorup = 'g', colordown = 'r', alpha = 0.8)
 
ax1.xaxis.set_major_formatter(dates.DateFormatter('%d/%m/%Y %H:%M'))
ax1.xaxis.set_major_locator(ticker.MaxNLocator(10))
 
plt.xticks(rotation = 30)
plt.grid()
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Historical Data')
plt.tight_layout()
plt.show()
 
 
 
 
 