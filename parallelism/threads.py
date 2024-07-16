import threading
import queue
from typing import *


class ProducerThread(threading.Thread):
    def __init__(self, q: queue.Queue, condition: threading.Condition, group: None = None, target: Callable[..., object] | None = ..., name: str | None = ..., args: Iterable[Any] = ..., kwargs: Mapping[str, Any] | None = ..., *, daemon: bool = True) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.q = q
        self.condition = condition
    
    def produce(self, data: any) -> None:
        self.condition.acquire()
        self.q.put(data)
        print("Put Data:", self.q.qsize())
        self.condition.notify_all()
        self.condition.release()
    
    def run(self) -> None:
        count = 1e3
        while count > 0:
            self.produce(count)
            count -= 1

class ConsumerThread(threading.Thread):
    def __init__(self, q:queue.Queue, condition: threading.Condition, func: Callable[[any], None], group: None = None, target: Callable[..., object] | None = ..., name: str | None = ..., args: Iterable[Any] = ..., kwargs: Mapping[str, Any] | None = ..., *, daemon: bool = True) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.q = q
        self.consumer_func = func
        self.condition = condition
    
    def consume(self) -> bool:
        self.condition.acquire()
        while self.q.qsize() == 0:
            self.condition.wait()
        self.consumer_func(self.q.get())
        self.condition.release()
        
        
    def run(self) -> None:
        while True:
            ret = self.consume()
