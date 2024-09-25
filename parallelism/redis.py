from typing import *
import jsonpickle
import redis
import threading

def redis_connecttion() -> redis.Redis:
    return redis.Redis(host='localhost', port=6379, db=0)

class RedisEncoder():
    """
    A class for encoding objects into JSON format using jsonpickle for Redis communication.
    """
    def encode(self):
        """
        Decodes the given JSON string into an object.

        Args:
            data (str): The JSON string to decode.

        Returns:
            object: The decoded object.
        """
        return jsonpickle.encode(self)

class RedisDecoder():
    """
    A class for decoding JSON strings into objects using jsonpickle for Redis communication.
    """
    def decode(self, data:str):
        """
        Decodes the given JSON string into an object.

        Args:
            data (str): The JSON string to decode.

        Returns:
            object: The decoded object.
        """
        return jsonpickle.decode(data)


class SyncRedisProducer():
    """
    A class for synchronous Redis message publishing.

    Attributes:
        conn (redis.Redis): The Redis connection object.
        channel (str | List[str]): The channel or list of channels to publish messages to.
    """
    def __init__(self, channel:str | List[str]):
        self.conn: redis.Redis = redis_connecttion()
        self.channel = channel
    
    def produce(self, data: RedisEncoder):
        """
        Publishes the encoded message to the Redis channel(s).

        Args:
            data (RedisEncoder): The data to publish.
        """
        json_data = data.encode()
        if isinstance(self.channel, list):
            for c in self.channel:
                self.conn.publish(c, json_data)
        else:
            self.conn.publish(self.channel, json_data)



class SyncRedisConsumer():
    """
    A class for synchronous Redis message consumption.

    Attributes:
        conn (redis.Redis): The Redis connection object.
        pubsub (redis.client.PubSub): The Redis pubsub object for message subscription.
    """
    def __init__(self, channel:str):
        """
        Initializes the SyncRedisConsumer with a Redis connection and subscribes to the specified channel.

        Args:
            channel (str): The Redis channel to subscribe to.
        """
        self.conn: redis.Redis = redis_connecttion()
        self.pubsub = self.conn.pubsub()
        self.pubsub.subscribe(channel)
        self.pubsub.ignore_subscribe_messages = True
    
    def convert_messages(self, message):
        """
        Converts the Redis message data into an object using RedisDecoder.

        Args:
            message (dict): The message received from Redis.

        Returns:
            object: The decoded message object.
        """
        rc = RedisDecoder()
        message_objs = rc.decode(message["data"])
        return message_objs
    
    def consume(self, messages)-> bool:
        """
        Placeholder for consuming messages. Should be implemented in subclasses.

        Args:
            messages (list): List of messages to consume.

        Returns:
            bool: Return True to continue, False to stop.
        """
        return


class RedisProducer(threading.Thread):
    """
    A class for asynchronous Redis message publishing using threading.

    Attributes:
        conn (redis.Redis): The Redis connection object.
        channel (str): The Redis channel to publish messages to.
    """
    def __init__(self, channel:str, group=None, target: Callable[..., object] | None = None, name: str | None = None, args: Iterable[Any] = None, kwargs: Mapping[str, Any] | None = None, *, daemon: bool | None = True) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.conn: redis.Redis = redis_connecttion()
        self.channel = channel
    
    def produce(self, data: RedisEncoder):
        """
        Publishes the encoded message to the Redis channel.

        Args:
            data (RedisEncoder): The data to publish.
        """
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
    """
    A class for asynchronous Redis message consumption from multiple channels using threading.

    Attributes:
        pubsubs (List[redis.client.PubSub]): List of Redis pubsub objects for message subscription.
    """
    def __init__(self, channels: List[str], group=None, target: Callable[..., object] | None = ..., name: str | None = ..., args: Iterable[Any] = ..., kwargs: Mapping[str, Any] | None = ..., *, daemon: bool = True) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.pubsubs = [redis_connecttion().pubsub() for i in range(len(channels))]
        for i in range(len(channels)):
            self.pubsubs[i].subscribe(channels[i])
            self.pubsubs[i].ignore_subscribe_messages = True
    
    def consume(self, messages) -> bool:
        """
        Consumes and processes Redis messages.

        Args:
            messages (list): List of messages received from Redis.

        Returns:
            bool: Returns True to keep consuming, False to stop.
        """
        objs = self.convert_messages(messages)
        return True
        
    
    def convert_messages(self, messages):
        """
        Converts a list of Redis messages into decoded objects.

        Args:
            messages (list): List of Redis messages.

        Returns:
            list: List of decoded message objects.
        """
        message_objs = [RedisDecoder().decode(message["data"]) for message in messages]
        return message_objs
        

    def run(self) -> None:
        """
        Starts listening to messages on multiple channels, converting and consuming them.
        """
        for m1, m2 in zip(self.pubsubs[0].listen(), self.pubsubs[1].listen()):
            transformed_messages = self.convert_messages([m1, m2])
            keep_listening = self.consume(transformed_messages)
            if not keep_listening:
                break
        print("RC consumer finished", keep_listening)


