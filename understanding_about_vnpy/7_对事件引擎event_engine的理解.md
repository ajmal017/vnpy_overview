# 对EventEngine的理解
1. _handlers = defaultdict(list)是成员变量,是一个字典,字典的键值是一个list
2. register函数,用于将事件追加到handler_list
3. 