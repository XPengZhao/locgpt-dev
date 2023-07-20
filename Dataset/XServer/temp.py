import yaml
from rabbitmq import MQConsumer

with open("conf.yaml") as f:
    kwargs = yaml.safe_load(f)
    f.close()

    try:
        mq = MQConsumer(**kwargs['mq'])
    except:
        raise Exception("can't connect to localhost")

mq.start_consuming()