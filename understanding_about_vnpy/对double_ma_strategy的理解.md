# 对vnpy 2.0的源码的理解
## 从double_ma_strategy作为入口
1. CtaTemplate是策略类的模板,每个策略都要继承后再重写相应的成员函数
2. 策略的成员变量parameters,variables是用于在UI中展示的变量
3. 策略类会调用父类CtaTemplate的构造函数进行初始化,也许是考虑到后续的多继承,才会使用super函数
4. 在BacktestingEngine中的add_strategy函数,根据import的strategy,然后调用策略类的__init__去初始化
这个策略类,然后实例会是BacktestingEngine的一个成员变量
5. BarGenerator()会根据提供的on_bar或者on_xmin_bar合成不同周期的bar,在调用的时候就是在on_tick中,做update_tick
因为bar是由tick合成的
6. ArrayManager()是用来处理计算指标时,需要处理的向量的类,主要是封装了TA-Lib中一些计算指标的函数,
后续可以自己进行扩充
7. ArrayManager的update_bar函数,通过提供的bar,对成员变量
    self.open_array
    
    self.high_array
    
    self.low_array
    
    self.close_array
    
    self.volume_array
    按照逐个新来的bar,更新向量的数值
8. ArrayManager的成员函数max,min,sma,std,cci,atr,rsi等等指标计算函数,通过调用TA-Lib的接口进行计算,并选择是返回向量还是数值
9. strategy的on_init中的load_bar函数从CtaTemplate继承下来,进而再策略类模板中,
调用的是cta_engine中的load_bar函数,此函数则是接受策略的on_bar函数,用on_bar函数处理从database或者rqdata拿到的原始数据
10. 总结一下,on_init--->load_bar--->load_bar(cta_engine)--->on_bar(此on_bar由具体策略决定),作为策略初始化的时候要初始化的一点数据
11. 策略中的put_event从template中继承来,调用ctaEngine中的put_strategy_event,收集一定的策略相关的数据,然后封装到
Event类中,打上EVENT_CTA_STRATEGY的标签,插入到时间引擎中,然后交给事件引擎做进一步处理
12. BarGenerator有update_tick把tick放进bar生成器,update_bar用来合成一分钟bar
13. ArrayManager的update_bar则是用合成好的bar,去更新成员变量中的array
14. ArrayManager调用一次update_bar,计数一次,当count>=100(默认是100),即是调入100根K线后
将ArrayManager标记为初始化完成inited=False



