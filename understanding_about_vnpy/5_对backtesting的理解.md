## 成员变量        
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
## 成员函数的作用
1. clear_data用于把成员变量先初始化**清空**，防止意外
2. set_parameters,初始化一部分成员变量，包括其实资金和开始结束日期
3. add_strategy，通过传入相应的策略类和该策略对应的setting字典，对策略进行实例化，然后
赋值给回测引擎的成员变量self.strategy，之后self.strategy就可以传入相关参数再调用之
4. load_data，由于是回测，所以只是从mongodb中提起相关的历史数据，然后转换为bar或者tick类型的数据
然后由peewee的ORM方法,转换为DbBarData或者DbTickData类型的数据,再调用各自的to_bar或者to_tick,转换为
bar或者tick,再存放到history_data中,作为回测的所有历史数据.
5. run_backtesting是回测开始的入口.首先根据用户确定的回测模式是bar还是tick,选择回测的function是
new_bar还是new_tick,然后策略初始化,会先从数据库加载10天(默认十天)的数据
6. run_backtesting中,strategy.on_init()会又调用load_bar(10),继而调用策略自身定义的on_bar函数,去ctaengine中提取数据
7. ctaengine则又调用策略中的on_bar函数对从数据库中拿到的bar函数进行处理,该多则多,该空则空(发出相应的订单)
8. **策略初始化完成的时候,其实已经跑了10个bar了??**
9. calculate_result用于计算所有的交易完成过后,所有交易的盈亏情况,首先判断trades是否为空,否则没有成交记录.
10. 从trades中遍历每个trade,从每个trade中提取成员变量datetime,根据每笔交易的时间,从daily_results
(从daily_results在update_daily_close中生成,update_daily_close在new_tick或者new_bar中调用)
提取每笔交易的结果,而daily_results数据结构在DailyResult中.然后把每笔交易的数据追加到daily_result的成员变量trades中,
这个成员变量是**每天**发生的交易记录的容器
11. calculate_pnl是DailyResult类的成员函数,用于计算每笔交易的盈亏,然后daily_result的各个成员变量中,total_pnl,net_pnl
等指标就计算好了
12. 计算完pnl之后,对pre_close和start_pos,用计算完的daily_result的成员变量进行更新
13. 待daily_results中所有的daily_result都计算完毕后,将daily_results转换为字典,再转换为Pandas DataFrame,便于再
进一步计算相关的统计指标
14. calculate_statistics采用向量化计算相关策略的性能指标，较容易理解，然后把所有的统计结果汇总到一个字典中，并返回
15. 得到calculate_statistics的返回值，方便在后续的参数优化
16. show_chart，单纯的把统计指标画出来，没有别的
17. **run_optimization**，用于策略的参数优化，需要用到optimization_setting类来生成参数组合，optimize函数来多次跑策略
18. generate_setting,把在优化类中的参数字典params中的所有参数做笛卡尔积，生成所有的参数组合，然后依次进行回测
19. pool = multiprocessing.Pool(multiprocessing.cpu_count())采用进程池，多核机器并行跑回测
20. pool.apply_async采用非阻塞的异步多进程，并提供相关参数给optimize函数，计算完后对参数进行排序输出，并返回计算结果
可以自行存储起来做进一步分析
21. update_daily_close讲bar的close_price,追加到daily_results中
22. new_bar和new_tick，会在跑回测的时候二选一，来处理bar或者tick数据，然后都会调用
cross_limit_order和cross_stop_order函数来发出相应的订单
23. cross_limit_order，首先用新来的bar或者tick对long_cross_price，short_cross_price，long_best_price，short_best_price进行
更新，然后遍历active_limit_orders中，尚未发出的，尚未成交的订单，调用策略中的on_order,并发出订单，然后从active_limit_orders
中pop出已经发出的订单，这时候，交易次数自增1
24. 然后，由空头和多头，赋值到策略类的仓位上
35. 完成的每笔交易，生成每个TradeData类，追加到trades中，方便后续统计
36. cross_stop_order与cross_limit_order大致差不多
37. load_bar，load_tick用于讲回测引擎的回调函数设置为用户自定义的on_bar或者on_tick,在runbacktesting中被调用
38. stop_orderid，也就是止损单号，由止损前缀STOP和订单数组成（**不过我觉得可以把数字换成是时间，这样具有唯一性**）
39. 然后把所有的止损单追加到active_stop_orders和stop_orders容器中
40. send_limit_order和send_stop_order基本一致
41. cancel_stop_order--撤销止损单，根据给定的订单号，先判断是否在active_stop_orders容器中，
如果存在则把这个订单从容器中用pop取出，讲订单的状态赋值为StopOrderStatus.CANCELLED，然后调用
策略的on_stop_order函数，撤销这个订单
42. cancel_limit_order和cancel_stop_order大致一致
43. cancel_all和cancel_order均用于取消所有的止损单
44. write_log，单纯写入日志到日志容器中
45. send_email发邮件
46. get_engine_type返回当前引擎的类型，默认是回测引擎
47. output,直接把相关的信息输出到控制台