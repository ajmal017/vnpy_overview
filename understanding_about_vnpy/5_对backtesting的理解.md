### 成员变量        
self.vt_symbol = ""  标的物代码，带交易所后缀

self.symbol = ""  标的物代码

self.exchange = None  交易所代码

self.start = None  回测开始日期

self.end = None  回测结束日期

self.rate = 0  交易手续费比例

self.slippage = 0  滑点设置

self.size = 1  交易几手

self.pricetick = 0  ???

self.capital = 1_000_00  初始资金量

self.mode = BacktestingMode.BAR  回测模式是按照bar来还是按照tick来

self.strategy_class = None  就是写好的策略类

self.strategy = None  策略的名字，要和策略类的名字一致

self.tick = None  用于迭代的tick数据

self.bar = None  用于迭代的bar数据

self.datetime = None  用于处理时间数据

self.interval = None  用于表示数据的频率，是分钟，还是小时等等

self.days = 0  天数

self.callback = None  回调函数，是bar还是tick

self.history_data = []  用于储存从数据库得到的原始数据

self.stop_order_count = 0  计数发送止损单的次数

self.stop_orders = {}  一个词典，用于容纳发出过的止损单

self.active_stop_orders = {}  用于容纳正在等待发出的止损单？

self.limit_order_count = 0  计数发送限价单的次数

self.limit_orders = {}  一个词典，用于容纳发出过的限价单

self.active_limit_orders = {}  用于容纳正在等待发出的限价单？

self.trade_count = 0  对交易次数计数，交易一次加一

self.trades = {}  容器，容纳所有发生过的交易

self.logs = []  容纳所有过程产生的日志

self.daily_results = {}  用于容纳每天的盈亏

self.daily_df = None  最终的回测结构，一个Pandas DataFrame

