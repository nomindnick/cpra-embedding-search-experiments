"""LLM evaluation module for assessing local LLMs on CPRA-relevant tasks."""

from .models import OllamaModel, list_available_models
from .tasks import Task, TaskType, TASKS
from .evaluator import LLMEvaluator, EvaluationResult

__all__ = [
    "OllamaModel",
    "list_available_models",
    "Task",
    "TaskType",
    "TASKS",
    "LLMEvaluator",
    "EvaluationResult",
]
