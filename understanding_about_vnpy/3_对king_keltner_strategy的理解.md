# 对vnpy 2.0的源码的理解
## KingKeltnerStrategy作为入口

1. 在on_5min_bar中，首先会取消所有在orderList中的订单
2. 取消订单通过继承自template的cancel_order函数
3. cancel_order函数继而调用CtaEngine的cancle_order函数，再进一步
调用cancel_stop_order和cancel_limit_order函数，取消所有的限价单和止损单
4. on_5min_bar--->cancel_order(类模板)--->cancle_order(ctaengine)--->cancel_stop_order&cancel_limit_order
5. buy,short,sell,cover函数，都会调用调用ctaengine的send_order
函数，然后返回一个订单号，再追加到成员变量orderList中
6. on_trade在cross_limit_order或者cross_stop_order中被调用，两个cross函数又在
new_bar或者在new_tick中被调用，new_bar或者new_tick在backterstingengine中
被赋值给func，用于历史回访bar或者tick的数据