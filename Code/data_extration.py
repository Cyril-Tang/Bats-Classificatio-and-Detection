import tushare as ts
from multiprocessing.dummy import Pool as ThreadPool
import time
from tqdm import tqdm



pro = ts.pro_api()

# Obtain  all stocks available
data = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
data = data.dropna()
data.drop(data[data.symbol.astype('int') > 604000].index, inplace=True)
data.industry = data.industry.astype('category')

stock_list = [i for i in data.ts_code]

def process_data(df):
    df.sort_index(axis=0, ascending = False, inplace = True)
    df.reset_index(drop=True, inplace=True)
    df = df.drop(['pre_close', 'change', 'amount'], axis=1)


def generate_data(stock):
    df = ts.pro_bar(ts_code=stock, adj='qfq', start_date=train_start , end_date=train_end)
    if df is not None:
        process_data(df)
        if len(df)>sample_length:
            df.to_csv('Data/train_raw/'+stock.split('.')[0]+'.csv')
            
    df1 = ts.pro_bar(ts_code=stock, adj='qfq', start_date=test_start , end_date=test_end)
    if df1 is not None:
        process_data(df1)
        if len(df1)>sample_length:
            df1.to_csv('Data/test_raw/'+stock.split('.')[0]+'.csv')


def main():
    pool = ThreadPool()
    list(tqdm(pool.imap(generate_data, stock_list), total=len(stock_list)))
    pool.close()
    pool.join()
    

if __name__ == '__main__':
    train_start = '2000-01-01'
    train_end = '2015-12-31'
    test_start = '2016-01-01'
    test_end = '2019-12-31'
    sample_length = 50
    start = time.process_time()
    main()
    print(time.process_time()-start)
