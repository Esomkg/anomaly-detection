import json
import logging
import smtplib
from email.mime.text import MIMEText
from typing import Any

logger = logging.getLogger(__name__)


class AlertNotifier:
    def __init__(self, config: dict):
        self.config = config.get("alerts", {})
        self.slack_webhook = self.config.get("slack_webhook_url")
        self.email_config = self.config.get("email", {})
        self.cooldown_seconds = self.config.get("cooldown_seconds", 60)
        self._last_alert_time: dict[str, float] = {}

    def notify(self, alert: dict, source: str = "anomaly-api") -> None:
        message = self._format_message(alert, source)

        if not self._check_cooldown(source):
            return

        logger.info("ALERT: %s", message)
        self._notify_console(message)

        if self.slack_webhook:
            self._notify_slack(message)

        if self.email_config.get("enabled"):
            self._notify_email(message)

    def _check_cooldown(self, source: str) -> bool:
        import time

        now = time.time()
        last = self._last_alert_time.get(source, 0)
        if now - last < self.cooldown_seconds:
            return False
        self._last_alert_time[source] = now
        return True

    def _format_message(self, alert: dict, source: str) -> str:
        timestamp = alert.get("timestamp", "unknown")
        score = alert.get("anomaly_score", 0)
        metrics = alert.get("metrics", {})
        cpu = metrics.get("cpu_pct", "?")
        mem = metrics.get("mem_pct", "?")
        latency = metrics.get("latency_ms", "?")
        error_rate = metrics.get("error_rate", "?")

        return (
            f"[{source}] Anomaly detected at {timestamp} | "
            f"Score: {score:.4f} | "
            f"CPU: {cpu}% | Mem: {mem}% | "
            f"Latency: {latency}ms | Errors: {error_rate}%"
        )

    def _notify_console(self, message: str) -> None:
        print(message)

    def _notify_slack(self, message: str) -> None:
        try:
            import requests

            payload = {"text": f":warning: {message}"}
            resp = requests.post(
                self.slack_webhook, data=json.dumps(payload), timeout=5,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                logger.error("Slack notification failed: %s", resp.text)
        except ImportError:
            logger.warning("requests not installed — skipping Slack notification")
        except Exception as e:
            logger.error("Slack notification error: %s", e)

    def _notify_email(self, message: str) -> None:
        cfg = self.email_config
        try:
            msg = MIMEText(message)
            msg["Subject"] = cfg.get("subject", "Anomaly Detection Alert")
            msg["From"] = cfg["from"]
            msg["To"] = cfg["to"]

            with smtplib.SMTP(cfg["smtp_host"], cfg.get("smtp_port", 587)) as server:
                server.starttls()
                server.login(cfg["smtp_user"], cfg["smtp_password"])
                server.send_message(msg)
        except Exception as e:
            logger.error("Email notification error: %s", e)


class CompositeNotifier:
    def __init__(self, config: dict):
        self.notifier = AlertNotifier(config)
        self.alert_history: list[dict] = []

    def process_prediction(self, record: dict, is_anomaly: bool, score: float) -> None:
        if not is_anomaly:
            return

        alert = {
            "timestamp": record.get("timestamp", ""),
            "metrics": record.get("metrics", {}),
            "anomaly_score": round(score, 4),
            "label": record.get("label"),
        }
        self.alert_history.append(alert)
        self.notifier.notify(alert)

    def get_summary(self) -> dict[str, Any]:
        if not self.alert_history:
            return {"total_alerts": 0}

        scores = [a["anomaly_score"] for a in self.alert_history]
        return {
            "total_alerts": len(self.alert_history),
            "avg_score": round(sum(scores) / len(scores), 4),
            "max_score": round(max(scores), 4),
        }
