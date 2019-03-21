# 对vnpy 2.0的源码的理解
## BollChannelStrategy作为入口
1. BarGenerator默认是合成一分钟的K线，提供一个整数，可以合成对应分钟数的x分钟K线
2. x分钟的K线则是从一分钟K线合成而来
3. update_tick调用由策略传入的on_bar函数,当前boll_channel_strategy
的on_bar函数又调用BarGenerator的update_bar函数
4. on_tick(策略类)--->update_tick（BarGenerator）--->on_bar（由策略类传递进来）
--->update_bar(BarGenerator)--->on_xmin_bar(BarGenerator)
5. on_tick在backtestingengine类中，被new_tick调用，new_tick又被run_backtesting调用