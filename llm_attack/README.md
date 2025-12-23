# LLM Attack Module

This module contains tools for **LLM-driven autonomous red teaming** to stress-test safety shields on Industrial Control Systems (ICS).

## Overview

The attack module uses Large Language Models (LLMs) as cognitive agents that adaptively attempt to bypass safety shields on a Siemens S7-1200 PLC testbed.

## Components

### `llm_red_team_agent.py`

The main attack agent that:
- Connects to a Siemens S7-1200 PLC via Snap7
- Uses multiple LLMs to generate adaptive attack strategies
- Attempts to overflow a water tank by manipulating setpoints
- Logs all attack attempts and outcomes for analysis

## Available LLMs

1. `openai/gpt-oss-120b`
2. `OpenGVLab/InternVL3_5-30B-A3B`
3. `Qwen/Qwen3-30B-A3B`
4. `openai/gpt-oss-20b`

## Usage

```bash
# List available models
python llm_red_team_agent.py --list-models

# Run with single model (mock PLC for testing)
python llm_red_team_agent.py --model "Qwen/Qwen3-32B"

# Run with all models for comparison
python llm_red_team_agent.py --all-models

# Run with real PLC hardware
python llm_red_team_agent.py --model "Qwen/Qwen3-32B" --real-plc

# Custom settings
python llm_red_team_agent.py --all-models --steps 100 --delay 1.0
```

## Configuration

Set environment variables in `.env` file:

```bash
# PLC Connection
PLC_IP=192.168.0.1
PLC_RACK=0
PLC_SLOT=1
DB_NUMBER=1

# Open WebUI LLM API
OPENWEBUI_URL=http://cci-siscluster1.charlotte.edu:8080/api/chat/completions
OPENWEBUI_API_KEY=your-api-key
LLM_MAX_TOKENS=2000
```

## Output

Results are saved to `llm_red_team_results.json` with:
- Per-step attack logs
- LLM reasoning traces
- Success/failure metrics
- Multi-model comparison summaries

## Research Purpose

This module is designed for **authorized HIL cybersecurity research** to evaluate the effectiveness of safety shields against adaptive AI-driven attacks.
