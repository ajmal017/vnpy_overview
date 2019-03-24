# 对EventEngine的理解
1. _handlers = defaultdict(list)是成员变量,是一个字典,字典的键值是一个list
2. register函数,用于将事件追加到handler_list
3. 


## 成员变量
    self._interval = interval  事件引擎更新频率,默认按照秒计算
    self._queue = Queue()  事件引擎的队列容器,用于存储所有有待处理的事件
    self._active = False   事件引擎启动的标识,启动时为True
    self._thread = Thread(target=self._run)  给_run函数一个独立的线程
    self._timer = Thread(target=self._run_timer)  给计时器又一个独立的线程
    self._handlers = defaultdict(list)  特定类型事件的容器,用于容纳将被处理的事件
    self._general_handlers = []  所有类型事件容器

## 成员函数
1. _run, 首先判断事件引擎是否被激活,从队列中拿到事件,然后调用_process函数,对这个事件进行处理
2. _process, 
3. _run_timer,一个计时器,每隔一个时间就把时间事件放入到队列中去处理
4. start,把事件引擎标志为启动,然后启动处理事件的线程,然后启动计时器的线程
5. stop, 把事件引擎标志为关闭,然后等待两个子线程的结束,然后主线程结束
6. put, 把事件放入事件队列中
7. register, 判断一个handler是否在handler_list中,如果不在,则追加进去
8. unregister, 同理,判断在handler_list容器中,在,则remove,如果容器为空,则从类型事件容器中删除这种类型的事件
9. register_general, 同理,对所有类型的事件都追加到通用类型事件容器中
10. unregister_general,从通用事件容器中移除掉制定的事件