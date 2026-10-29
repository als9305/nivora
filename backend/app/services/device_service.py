import hashlib
from typing import Dict, Any, Optional

class DeviceService:
    @staticmethod
    def calculate_device_risk(device_info: Optional[Dict[str, Any]], user_history: Optional[list] = None) -> float:
        """
        Calculates non-sensitive device and session risk score (0 - 100).
        """
        if not device_info:
            return 15.0

        risk = 10.0  # baseline safe device risk

        user_agent = device_info.get("user_agent", "")
        ip = device_info.get("ip", "127.0.0.1")
        is_new_device = device_info.get("is_new_device", False)
        rapid_attempts = device_info.get("rapid_attempts", False)

        if is_new_device:
            risk += 25.0

        if rapid_attempts:
            risk += 35.0

        # Check for headless browser / automated proxy indicators in User Agent
        ua_lower = user_agent.lower()
        if "puppeteer" in ua_lower or "selenium" in ua_lower or "headless" in ua_lower:
            risk += 50.0

        return float(min(100.0, max(0.0, risk)))
