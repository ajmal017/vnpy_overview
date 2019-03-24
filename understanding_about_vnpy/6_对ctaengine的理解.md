# 对CtaEngine的理解
1. engine_type = EngineType.LIVE，CtaEngine可用于实盘交易
2. 没有找到ctaengine实例化的地方?????
3. CtaEngine继承自BaseEngine,BaseEngine则包括是主引擎和事件引擎和引擎名字三大要素,算是一个基础接口 
4. CtaEngine通过主引擎MainEngine和事件引擎EvnetEngine作为参数实例化
5. 主引擎MainEngine则又通过事件引擎EnventEngine实例化
## 成员变量
    self.strategy_setting = {}  用于容纳各种策略的各种参数,用于后续的策略加载
    self.strategy_data = {}  用于存储需要保存的策略的数据
    
    self.classes = {}  用于存储多个策略的类
    self.strategies = {}  用于存储多个策略的名字
    
    self.symbol_strategy_map = defaultdict(list)  
    self.orderid_strategy_map = {}  
    self.strategy_orderid_map = defaultdict(set)  
    
    self.stop_order_count = 0  
    self.stop_orders = {}  
    
    self.init_thread = None
    self.init_queue = Queue()
    
    self.rq_client = None
    self.rq_symbols = set()

## 成员函数
1. init_engine是个比较高级的函数,后续再说
2. register_event将三个对tick,order,trade处理的函数,追加到handler_list中
3. init_rqdata用于初始化米筐的数据接口
4. query_bar_from_rq,从米筐拿到DataFrame,然后按照迭代的方法,自己合成指定分钟数的K线
字段为open,high,low,close,wolume
5. process_tick_event
6. process_order_event
7. process_trade_event
8. check_stop_order,检查根据新来的tick数据是否需要发出止损单.  首先判断新来的tick的标的物代码是否在现有的止损单
容易中,不存在就算了.存在的情况下,判断止损单的多空头和相关的价格与新来的tick的价格的关系,再进一步判断是否需要处理
9. 如果long或者short被触发,先得到相应的止损单对应的策略,然后根据做单的方向,再根据tick的多档行情决定需要做单的价格,
然后通过send_limit_order发送相应的限价单,发送限价单后,再从stop_orders和vt_orderids两个容器中中去掉相应的订单和订单号
10. call_strategy_func可以用于调用策略中用户自定义的各种函数,方便用户实现更灵活的需求
11. send_limit_order,首先通过传入的策略,然后根绝策略的标的物代码,通过get_contract函数获得该标的物所有的合约
12. 然后根据字典ORDER_CTA2VT获得相应映射关系的做单方向direction和偏移量offset,然后生成通过dataclas OrderRequest生成
相关的订单请求.然后就通过相关券商的底层接口发送出去.
13. 然后会把策略和相关的订单号,追加到所有的订单列表中.
14. send_stop_order则和上面个的limit差不多,但是订单是通过策略的on_stop_order实现的
15. cancel_limit_order和cancel_stop_order则又和上面两个函数类似,通过容器获取相关信息,然后传递给mainengine或者通过
策略的回调函数进行发送请求.
16. send_order,cancel_order,cancel_all三个函数,则可以按照一定的条件对以上四个函数进行组合使用.
17. load_bar和load_tick则是在之前已经说过,用于从米筐或者从mongodb中获取数据,然后由策略中的回调函数进行处理
18. call_strategy_func作为一个结构,用于调用策略中用户编写的自定义函数,比较方便
19. add_strategy,首先检查新添加的策略是否个现有策略名字重复冲突,然后将其实例化后追加到策略容器strategies中,然后更新
具体策略的设定,并讲该策略的各项数据添加到事件引擎中
20. init_strategy,会检查线程是否存在,然后为_init_strategy函数,创建一个新的线程,这个线程会执行具体策略的on_init函数,并且
通过底层接口去订阅策略中的行情数据.
21. start_strategy,stop_strategy两个函数则是会调用策略本身的on_start,on_stop两个函数,去执行特殊的功能
22. edit_strategy,用提供的setting去更新strategy的策略
23. remove_strategy,从策略容器strategies中筛除某个策略,并且删除相关策略的策略,并且**删除与这个策略相关的订单**
24. load_strategy_class--->load_strategy_class_from_folder--->load_strategy_class_from_module(层层嵌套),
从.py文件load策略,然后追加到classe中
25. load_strategy_data即是从文件中读取数据保存到strategy_data容器中
26. sync_strategy_data则是从现有的内存中,把数据再同步到磁盘上
27. get_all_strategy_class_names,返回classes容器中,所有的策略的名字
28. get_strategy_class_parameters,返回值为策略中的各种参数
29. get_strategy_parameters返回某个策略的参数
30. init_all_strategies,遍历策略容器,调用前面的init_strategy函数,也是调用相应策略的on_init函数,初始化所有的策略
31. start_all_strategies,遍历策略容器,调用前面的start_strategy函数,也是调用相应策略的on_start函数,开始所有的策略
32. stop_all_strategies,遍历策略容器,调用前面的stop_strategy函数,也是调用相应策略的on_stop函数,开始所有的策略
33. load_strategy_setting,从策略文件所在路径中,读取所有的策略,然后调用add_strategy函数来加载策略的各项参数
34. update_strategy_setting,则是把内存中的策略设置保存到磁盘中
35. remove_strategy_setting,从策略设置容器中,删除指定策略的设定,然后再保存到文件中,完成对先前文件的覆盖,也就是完成了更新
36. put_stop_order_event,把一个stop_order放到事件引擎中
37. put_strategy_event,把一个策略数据,放到事件引擎中
38. write_log,根据策略的名字,加上相应的消息,合成为LogData类型的日志数据,再添加到事件引擎
39. send_email,发送电子邮件系列
