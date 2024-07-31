from typing import *
import jsonpickle
import redis
import threading

def redis_connecttion() -> redis.Redis:
    return redis.Redis(host='localhost', port=6379, db=0)

class RedisEncoder():
    def encode(self):
        return jsonpickle.encode(self)

class RedisDecoder():
    def decode(self, data:str):
        return jsonpickle.decode(data)


class SyncRedisProducer():
    def __init__(self, channel:str | List[str]):
        self.conn: redis.Redis = redis_connecttion()
        self.channel = channel
    
    def produce(self, data: RedisEncoder):
        json_data = data.encode()
        if isinstance(self.channel, list):
            for c in self.channel:
                self.conn.publish(c, json_data)
        else:
            self.conn.publish(self.channel, json_data)



class SyncRedisConsumer() :
    def __init__(self, channel:str):
        self.conn: redis.Redis = redis_connecttion()
        self.pubsub = self.conn.pubsub()
        self.pubsub.subscribe(channel)
        self.pubsub.ignore_subscribe_messages = True
    
    def convert_messages(self, message):
        rc = RedisDecoder()
        message_objs = rc.decode(message["data"])
        return message_objs
    
    def consume(self, messages)-> bool:
        return


class RedisProducer(threading.Thread):
    def __init__(self, channel:str, group=None, target: Callable[..., object] | None = None, name: str | None = None, args: Iterable[Any] = None, kwargs: Mapping[str, Any] | None = None, *, daemon: bool | None = True) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.conn: redis.Redis = redis_connecttion()
        self.channel = channel
    
    def produce(self, data: RedisEncoder):
        json_data = data.encode()
        if isinstance(self.channel, list):
            for c in self.channel:
                self.conn.publish(c, json_data)
        else:
            self.conn.publish(self.channel, json_data)


def consume(messages: List[any]) -> Tuple[bool, RedisDecoder]:
    message_type = "message" if [message["type"] for message in messages].count("message") == len(messages) else "diff_message"
    if message_type == "message":
        for message in messages:
            data = message["data"]
            decoded_data = data.decode('utf-8')
            decoder = RedisDecoder()
            obj = decoder.decode(decoded_data)
        if obj.terminate:
            return False
    return True



class RedisConsumer(threading.Thread):
    def __init__(self, channels: List[str], group=None, target: Callable[..., object] | None = ..., name: str | None = ..., args: Iterable[Any] = ..., kwargs: Mapping[str, Any] | None = ..., *, daemon: bool = True) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.pubsubs = [redis_connecttion().pubsub() for i in range(len(channels))]
        for i in range(len(channels)):
            self.pubsubs[i].subscribe(channels[i])
            self.pubsubs[i].ignore_subscribe_messages = True
    
    def consume(self, messages) -> bool:
        objs = self.convert_messages(messages)
        return True
        
    
    def convert_messages(self, messages):
        message_objs = [RedisDecoder().decode(message["data"]) for message in messages]
        return message_objs
        

    def run(self) -> None:
        for m1, m2 in zip(self.pubsubs[0].listen(), self.pubsubs[1].listen()):
            transformed_messages = self.convert_messages([m1, m2])
            keep_listening = self.consume(transformed_messages)
            if not keep_listening:
                break
        print("RC consumer finished", keep_listening)


