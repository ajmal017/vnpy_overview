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


class Lyn_Knn_Strategy(CtaTemplate):
    """this strategy is based on the KNN ML ALGO"""
    author = 'lyn'

    num_of_k = 5
    look_back = 5  # how much data we used in the past
    ma_window = 5  # 暂定是5个window,用于计算收盘价的乖离度

    parameters = ['num_of_k']
    variables = ['']

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
