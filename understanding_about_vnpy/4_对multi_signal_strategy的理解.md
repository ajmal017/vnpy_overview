# 对vnpy 2.0的源码的理解
## MultiSignalStrategy作为入口

1. 在这个多信号策略中，通过多个子策略来生成最终的交易信号
2. 三个子策略分别是RsiSignal，CciSignal，MaSignal，采用不同的算法分别生成多头，空头，和不动的仓位
3. 通过继承自基类CtaSignal的set_signal_pos函数，来设置成员变量signal_pos的值
分别是-1，+1，0，代表空头，多头和不动
4. MultiSignalStrategy则继承自TargetPosTemplate类
5. 三个RSI,CCI,MA策略分别调用on_tick和on_bar函数处理数据
6. 在calculate_target_pos函数中，分别获取三个策略的多空仓位值，然后进行投票，最终确定多信号策略的仓位值
