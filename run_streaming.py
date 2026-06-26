import argparse


def run_producer():
    from src.streaming.producer import MetricsProducer
    producer = MetricsProducer("configs/streaming.yaml")
    try:
        producer.run(delay=0.02)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        producer.close()


def run_consumer():
    from src.streaming.consumer import MetricsConsumer
    consumer = MetricsConsumer("configs/streaming.yaml")
    try:
        alerts = consumer.run(max_records=500)
        print(f"Total alerts: {len(alerts)}")
    except KeyboardInterrupt:
        print("\nStopped by user")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly Detection Streaming")
    parser.add_argument("mode", choices=["producer", "consumer"], help="Run mode")
    args = parser.parse_args()

    if args.mode == "producer":
        run_producer()
    else:
        run_consumer()
