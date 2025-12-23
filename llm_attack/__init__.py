"""
LLM Attack Module for HAI Testbed
=================================

This module contains tools for LLM-driven autonomous red teaming
to stress-test safety shields on ICS systems.

Components:
    - llm_red_team_agent.py: Main attack agent using multiple LLMs
"""

from .llm_red_team_agent import (
    RedTeamAgent,
    PLCConnection,
    MockPLCConnection,
    LLMClient,
    CampaignResult,
    AttackResult,
    AVAILABLE_MODELS,
    run_multi_model_comparison,
)

__all__ = [
    'RedTeamAgent',
    'PLCConnection', 
    'MockPLCConnection',
    'LLMClient',
    'CampaignResult',
    'AttackResult',
    'AVAILABLE_MODELS',
    'run_multi_model_comparison',
]
