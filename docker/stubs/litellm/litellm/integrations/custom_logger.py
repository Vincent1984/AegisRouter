"""Minimal CustomLogger stub for aegis_router callbacks."""
from typing import Literal, Union, Optional


class CustomLogger:
    """Custom callback logger base class used by aegis_router."""

    def __init__(self):
        pass

    def log_pre_api_call(self, model, messages, kwargs):
        pass

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_pre_api_call(self, model, messages, kwargs):
        pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_pre_call_check(self, deployment: dict) -> Optional[dict]:
        pass

    def pre_call_check(self, deployment: dict) -> Optional[dict]:
        pass

    async def async_pre_call_hook(self, user_api_key_dict, cache, data: dict, call_type):
        pass

    async def async_post_call_failure_hook(self, original_exception, user_api_key_dict):
        pass

    async def async_post_call_success_hook(self, user_api_key_dict, response):
        pass

    async def async_post_call_streaming_hook(self, user_api_key_dict, response: str):
        pass
