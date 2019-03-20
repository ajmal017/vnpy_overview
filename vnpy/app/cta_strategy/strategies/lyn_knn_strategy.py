from vnpy.app.cta_strategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)
import numpy as np
import pandas as pd

class Lyn_Knn_Strategy(CtaTemplate):
    """this strategy is based on the KNN ML ALGO"""
    author = 'lyn'

    num_of_knn = 5 # default num of the K nearst neighbor
    look_back = 5  # how much data we used in the past
    ma_window = 5  # 暂定是5个window,用于计算收盘价的乖离度

    parameters = ['num_of_k']
    variables = ['predict_return']

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super(Lyn_Knn_Strategy, self).__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager()

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.load_bar(10)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")
        self.put_event()

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

        self.put_event()

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        self.bg.update_tick(tick)

    # 仿照按照on_bar,而不是按照tick
    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """

        am = self.am
        am.update_bar(bar)
        if not am.inited:
            return

        # am.close是一个bar中的一个有close_price组成的array，并
        # 在每次update_bar的时候，会变长
        # array=True会返回一个向量,否则为数值
        # bar.close返回一个数值
        ma_value = am.sma(self.ma_window, array=False)
        close_bias = bar.close_price - ma_value
        close_bias_norm = close_bias / (am.max(self.ma_window, array=False)
                                        - am.min(self.ma_window, array=False))

        volume_ma_value = am.sma(self.ma_window, bar_component='volume',array=False)
        volume_bias = bar.volume - volume_ma_value
        close_bias_norm = volume_bias / (am.max(self.ma_window,bar_component='volume', array=False)
                                        - am.min(self.ma_window, bar_component='volume',array=False))

        combination_matrix = np.array((close_bias_norm,close_bias_norm))
        past_market_data = pd.DataFrame(np.array([1,2,3]))
        all_dist = np.sqrt(np.sum(np.square( past_market_data.values - combination_matrix), axis=1))
        dist_index = all_dist.argsort()[:self.num_of_knn]
        # ----------------------------------------------------
        predict_return = past_market_data.iloc[dist_index]['future_return'].mean()
        # ----------------------------------------------------
    #     future return is the target variable of the predicting

        if predict_return >= 0.03:
            if self.pos == 0:
                self.buy(bar.close_price, 1)
            elif self.pos < 0:
                self.cover(bar.close_price, 1)
                self.buy(bar.close_price, 1)

        elif predict_return <= -0.03:
            if self.pos == 0:
                self.short(bar.close_price, 1)
            elif self.pos > 0:
                self.sell(bar.close_price, 1)
                self.short(bar.close_price, 1)

        self.put_event()

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        pass

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass