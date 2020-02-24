# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 00:53:33 2020

@author: Cyril
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_finance as mpf
import os
from tqdm import tqdm
import multiprocessing
import time
import sys
import datetime

path = "Data/"
length = 50


def new_file(testdir):
    #列出目录下所有的文件
    list = os.listdir(testdir)
    #对文件修改时间进行升序排列
    list.sort(key=lambda fn:os.path.getmtime(testdir+'\\'+fn))
    #获取最新修改时间的文件
    filetime = datetime.datetime.fromtimestamp(os.path.getmtime(testdir+list[-1]))
    #获取文件所在目录
    filepath = os.path.join(testdir,list[-1])
    # print("最新修改的文件(夹)："+list[-1])
    # print("时间："+filetime.strftime('%Y-%m-%d %H-%M-%S'))
    return filepath

# last_file = new_file('Data/up_2/')
# print('last file name is:', last_file)
# last_stock = last_file.split('_')[0]
# last_stock = '000910'


def generate_img_train(file):
    data = pd.read_csv(path+'train_raw/'+file, index_col = 0)
    data_copy = data[:-length].copy()
    print('train img working on:' + file)
    
    # if int(data.ts_code[0].split('.')[0]) < int(last_stock):
    #     return
    for i in (range(len(data)-length)):
        
        start = i
        img_name = data.ts_code[start].split('.')[0]+'_'+str(data_copy.trade_date[start])+'.png'
        # print('continue with stock:', data.ts_code[0].split('.')[0])
        # data_copy.drop([start], inplace=True)
        img_data = data[start:start+length]
        
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_axes([0,0.2,1,0.8])
        ax2 = fig.add_axes([0,0,1,0.2])
        
#         ax.set_xticks(range(0, len(img_data['trade_date']), 10))
#         ax.set_xticklabels(img_data['trade_date'][::10])
        
        mpf.candlestick2_ochl(ax, img_data['open'], img_data['close'], img_data['high'], img_data['low'],
                            width=0.5, colorup='r', colordown='green',
                            alpha=1)
#        plt.xticks([])
#        plt.yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(False)

        mpf.volume_overlay(ax2, img_data['open'], img_data['close'], img_data['vol'], colorup='r', colordown='g', width=0.5, alpha=0.8)
#        plt.xticks([])
#        plt.yticks([])
        ax2.spines['top'].set_visible(True)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        plt.subplots_adjust(hspace=0)
        
        if data['pct_chg'][start+length] > 2:
            plt.savefig("Data/up/"+img_name)
        if 0 < data['pct_chg'][start+length] <= 2:
            plt.savefig("Data/up_2/"+img_name)
        if -2 < data['pct_chg'][start+length] <= 0:
            plt.savefig("Data/down_2/"+img_name)
        if data['pct_chg'][start+length] <= -2:
            plt.savefig("Data/down/"+img_name)
        plt.close(fig)


def generate_img_test(file):
    data = pd.read_csv(path+'test_raw/'+file, index_col = 0)
    data_copy = data[:-length].copy()
    print('test img working on:' + file)
    for i in (range(len(data)-length)):
        start = i
        img_name = data.ts_code[start].split('.')[0]+'_'+str(data_copy.trade_date[start])+'.png'
        # data_copy.drop([start], inplace=True)
        img_data = data[start:start+length]
        
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_axes([0,0.2,1,0.8])
        ax2 = fig.add_axes([0,0,1,0.2])
        
#         ax.set_xticks(range(0, len(img_data['trade_date']), 10))
#         ax.set_xticklabels(img_data['trade_date'][::10])
        
        mpf.candlestick2_ochl(ax, img_data['open'], img_data['close'], img_data['high'], img_data['low'],
                            width=0.5, colorup='r', colordown='green',
                            alpha=1)
#        plt.xticks([])
#        plt.yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(False)

        mpf.volume_overlay(ax2, img_data['open'], img_data['close'], img_data['vol'], colorup='r', colordown='g', width=0.5, alpha=0.8)
#        plt.xticks([])
#        plt.yticks([])
        ax2.spines['top'].set_visible(True)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        plt.subplots_adjust(hspace=0)
        
        plt.savefig("Data/test/"+img_name)
        plt.close(fig)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                   
        os.makedirs(path)            
        print( "---  new folder...  ---")
        print( "---  OK  ---")

    else:
        print ("---  There is this folder!  ---")



def main():
    
    mkdir(path+'up')
    mkdir(path+'up_2')
    mkdir(path+'down')
    mkdir(path+'down_2')
    mkdir(path+'test')
    cores = multiprocessing.cpu_count()
    print("Instance cores:", cores)
    pool = multiprocessing.Pool(processes=cores)
    
    # train_files= os.listdir('Data/train_raw') 
    test_files = os.listdir('Data/test_raw')

    # cnt = 0
    # for _ in pool.imap_unordered(generate_img_train, train_files):
    #     sys.stdout.write('done %d/%d\r' % (cnt, len(train_files)))
    #     cnt += 1

    cnt1 = 0
    for _ in pool.imap_unordered(generate_img_test, test_files):
        sys.stdout.write('done %d/%d\r' % (cnt1, len(test_files)))
        cnt1 += 1

    # for file1 in test_files:
    #     print('test img working on:' + file1)
    #     generate_img_test(file1)
    pool.terminate()
    pool.join()

if __name__ == '__main__':
    
    start = time.process_time()
    
    main()
    
    
        
    print(time.process_time()-start)
        
        
        
        
        
        
        
