import numpy as np
import yfinance as yf
from typing import List
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
"""

"""



class Simulate_FCN:
    def __init__(self, stock_list: List[str], start_date: str, end_date: str):
        
        """
        初始化 輸入標地：台灣股票代碼
        起始日期
        結束日期
        直接抓資料
        """
        
      
        if not isinstance(stock_list, list) or not all(isinstance(t, str) for t in stock_list):
            raise TypeError("symbols_list 必須是包含標的名稱（字串）的列表 ")

        date_format = "%Y-%m-%d"
        try:
            datetime.strptime(start_date, date_format)
            datetime.strptime(end_date, date_format)
        except ValueError:
            raise ValueError("start_date 和 end_date 必須為 'YYYY-MM-DD' 格式的字串")
        
        self.__stock_list = stock_list
        self.__start_date = start_date
        self.__end_date = end_date
        self.__fetch_data()
            
        
    def __fetch_data(self)-> pd.DataFrame:
        """
        抓資料使用yahoo finance  抓調整後價格 
        TW 是上市,先預設是上櫃 可以改成直接輸入yf的代碼
        TWO 是上櫃
        """
        
        tickers = [s+".TW" for s in self.__stock_list]
        data = yf.download(tickers, start=self.__start_date, end=self.__end_date)['Adj Close']
        
        
        self.__log_return = np.log(data/data.shift(1))[1:]
        self.__sigma = self.__log_return.std()#日標準差
        self.__historical_data = data
        self.__corr_matrix = self.__log_return.corr()
        self.__cholesky_matrix = np.linalg.cholesky(self.__corr_matrix)
                        
        
    def pricing(self,**params)->float: #輸出現值
        """
        包括三步驟
        1. 模擬股價 simulate_prices
        2. 選擇權端報酬計算 discount_cashflows
        3. 加入固定端報酬計算 fixed_coupon
        
        4.是否要畫報酬圖
        """
        default_params = {
            'S0': 100, #股價漲跌買賣是看相對價格 S0: 報價單位
            'K0': 100,        
            'K1': 60,        
            'K': 70,
            
            'method': "GBM",
            'q': 0,        
            'r': 0.01,        
            's_sigma': 0,
            's_rho':0,
            'TENOR': 6,
            'nSim' : 1e4,
            'beta' : 2,
            
            
            "coupon": 0.02
            
        }
        default_params.update(params)
        
        
        self.__method = default_params["method"]
        sim_keys = ["S0","q" , "r" ,"s_sigma" , "s_rho", "TENOR","nSim","beta"]
        simu_params = {key: default_params[key] for key in sim_keys}
        self.__simulate_prices(method = self.__method, **simu_params)
        
       
        
        discount_keys = ["K0", "K1", "K" ]
        discount_params = {key: default_params[key] for key in discount_keys}
        self.__discount_cashflows(**discount_params)
        
        
        
        coupon_keys = ["coupon"]
        coupon_params = {key: default_params[key] for key in coupon_keys}
        self.__fixed_coupon(**coupon_params)
        
        
        
        


        return np.mean(self.__PV)                           

    def __simulate_prices(self, method: str="GBM", **params):
        
        """
        檢查方法是否是目前可使用的且方法必選
        """
        valid_methods = ["GBM", "CEV"]
        if method not in valid_methods:
            raise ValueError(f"無效的方法: '{method}'。有效選項包含: {', '.join(valid_methods)}")
        
        """
        可外部更動的參數
        TENOR 輸入單位為月
        """
        default_params = {
            'S0': 100,
            'q': 0,        
            'r': 0.01,        
            's_sigma': 0,
            's_rho':0,
            'TENOR': 6,
            'nSim' : 1e4,
            'beta' : 2
        }
        
        default_params.update(params)
        
        
        
        """
        模擬需要固定參數 不可更動
        """
        N =  len(self.__stock_list)
        s_sigma = default_params["s_sigma"]
        self.__S0 = default_params["S0"]
        S0 = np.ones(N)* self.__S0
        q = np.zeros(N)+ default_params["q"]
        sigma = [i *(1+s_sigma)*np.sqrt(251) for i in self.__sigma] #日標準差換成年標準差 *上sqrt(251)
        
        self.__r = default_params["r"]+default_params["s_rho"]
        self.__TENOR =  default_params["TENOR"]
        dt = 1/12
        self.__nSim = int(default_params['nSim'])
        
        
        
        
        """
        生成常態隨機亂數
        """
        
        
        rng = np.random.default_rng()
        '''
        模擬方法有點爛
        有空改成模擬該標的才生成常態
        不用存那麼大的變數
        '''
        
        normal_matrix = np.stack([ rng.standard_normal((self.__nSim,self.__TENOR)) for _ in range(N) ] , axis = -1)
        
        correlated_normal_matrix = normal_matrix @ self.__cholesky_matrix.T
        self.__result = np.zeros((N,self.__nSim, self.__TENOR+1))
        
       
        if method == "GBM":
            for n in range(N):
                for iSim in range(self.__nSim):
                    self.__result[n, iSim, 0] = S0[n]
                    for iStep in range(self.__TENOR): 
                        self.__result[n, iSim, iStep+1] = self.__result[n, iSim,iStep] * np.exp((self.__r - q[n] - 0.5 * sigma[n] ** 2) * dt + sigma[n] * np.sqrt(dt) * correlated_normal_matrix[iSim,iStep,n])
        
        
        elif method == "CEV":
            self.__bata = default_params['beta']
            for n in range(N):
                for iSim in range(self.__nSim):
                    self.__result[n, iSim, 0] = S0[n]
                    for iStep in range(self.__TENOR): 
                        self.__result[n, iSim, iStep+1] = (self.__result[n, iSim,iStep] * np.exp(
                            (self.__r - q[n] - 0.5 * self.__result[n, iSim,iStep]** (2 * self.__beta-1 )* sigma[n] ** 2) * dt 
                            + sigma[n] * self.__result[n, iSim,iStep]**(self.__bata -1 ) * np.sqrt(dt) * correlated_normal_matrix[iSim,iStep,n])
                            )        
        
    def __discount_cashflows(self, **params):
        """
        需要輸入參數 : 
        提前出場機制 K0: 
        下限價（觸及生效價格）K1
        賣出put執行價: K
        -------
        根據不同情境進行選擇權端報酬計算
        需要回傳每一次模擬的報酬 供pricing 計算
        """
        
        default_params={
            'K0':100,
            'K1':60,
            'K':70
            }
        
        
        default_params.update(params)
        
        self.__K0 = default_params["K0"]
        self.__K1 = default_params["K1"]
        self.__K = default_params["K"]
        
        
        """
        
        提前解約部分
        ---------------------
        period 每次模擬契約時長
        is K0 是否有提前解約
        
        
        """
        self.__period = np.zeros(self.__nSim, dtype='int64') 
        self.__isK0 = np.zeros(self.__nSim) 
        self.__scenario = np.zeros(self.__nSim, dtype='int64')
        self.__PV = np.zeros(self.__nSim)
        
        N = len(self.__stock_list)
        dt = 1/12
        payoff = np.zeros(self.__nSim)
        
        
        for iSim in range(self.__nSim):
            self.__period[iSim] = self.__TENOR
            period_t = 0 
            isK0_t = 0
            
            for n in range(N):
                
                for iStep in range(self.__TENOR):
                    if self.__result[n,iSim,iStep + 1 ] >= self.__K0:
                        
                        self.__period[iSim] = max(iStep+1 , period_t)
                        period_t = self.__period[iSim]
                        isK0_t = isK0_t +1
                        break
            
            
            
            self.__PV[iSim] = self.__S0 * np.exp(- self.__r * self.__period[iSim] * dt)
            
            if isK0_t == N:
                self.__isK0[iSim] = True
                self.__scenario[iSim] = 1
                continue
            
            if min(self.__result[:,iSim,self.__TENOR]) >= self.__K:
                if np.min(self.__result[:,iSim, :]) >= self.__K:
                        self.__scenario[iSim] = 3
                        continue
                self.__scenario[iSim] = 2
                continue

            
            # 生效且低於履約價，需要執行賣權
            
            self.__scenario[iSim] = 4
            payoff[iSim] = min(self.__result[:,iSim,self.__TENOR]) - self.__K
            self.__PV[iSim] = self.__PV[iSim] + payoff[iSim] * np.exp(- self.__r * self.__TENOR * dt)      
            

            
    def __fixed_coupon(self, **params) -> float:
        
        
        
        default_params={
            'coupon':0.02,
        }
        default_params.update(params)
        
        
        self.__coupon_rate = default_params["coupon"]
        dt = 1/12
        
        for iSim in range(self.__nSim):
            for iStep in range(self.__period[iSim]):
                self.__PV[iSim] = self.__PV[iSim] +  self.__S0 * self.__coupon_rate * np.exp(- self.__r * (iStep+1) * dt)
        
        
        
    
    
        


    
    
    
    
    """
    索引私人attribute
    """
    def get_return(self):
        
        return self.__log_return
    def get_method(self):
        
        return self.__method
    
    def get_sigma(self):
        return self.__sigma*np.sqrt(251)
    
    def get_r(self):
        return self.__r
    
    def get_TENOR(self):
        return self.__TENOR
    
    def get_historical_data(self):
        return self.__historical_data
    
    def get_cholesky_matrix (self):
        
        return self.__cholesky_matrix
    def get_corr_matrix(self):
        return self.__corr_matrix
    
    def get_beta_cev(self):
        
        return self.__beta
    
    def get_simulate_stock_result(self):
        
        return self.__result
    
    def get_PV(self):
        
        return self.__PV
    def get_K0(self):
        
        return self.__K0
    
    def get_K1(self):
        
        return self.__K1
    
    def get_strike(self):
        
        return self.__K
    
    
    def get_coupon(self):
        
        return self.__coupon_rat
    
    def get_VaR(self, alpha: float = 0.95):
        
        #假設以訂價發行
        
        
        profit = np.sort(self.__PV)
        varIndex = np.int64(self.__nSim * (1-alpha))
        varProfit = profit[varIndex] #- np.mean(profit)
        esProfit = np.mean(profit[0 : varIndex]) #- np.mean(profit)
        print('VaR:', varProfit)
        print('Expected Shortfall', esProfit)
        
        return esProfit,esProfit
        
    def get_scenario(self, plot = True):
        if plot == True:


            
            data = self.__scenario
            unique, counts = np.unique(data, return_counts=True)
            proportions = counts / counts.sum()
            labels = [f"Scenario {val}" for val in unique]

            plt.figure(figsize=(6, 6))
            plt.pie(proportions, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab10.colors[:len(unique)])
            plt.title("Proportion of Scenario")
            plt.savefig("Scenario.png",transparent=True )
            plt.show()

            return 0
        else: 
            return self.__scenario

if __name__ =='__main__':
    stocks_list = ["2330", "2317" , "2454" , "2881", "2382"]
    start = '2024-05-01'
    end='2024-11-01'
    
    FCN = Simulate_FCN(stocks_list, start,end)
    data = FCN.get_historical_data()
    default_params = {
            'q': 0,        
            'r': 0.01,        
            's_sigma': 0,
            's_rho':0,
            'TENOR': 6,
            'nSim' : 1e4,
            'beta' : 2,
            "K0":100
            
        }
    
    price = FCN.pricing(**default_params)
    result = FCN.get_simulate_stock_result()
    PV = FCN.get_PV()
    VaR , ES = FCN.get_VaR(alpha = 0.99)
    FCN.get_scenario(plot = True)
    print(price)
    



    
