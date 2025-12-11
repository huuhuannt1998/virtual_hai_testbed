"""
Safety Shield for HAI-ML
========================

Rule-based safety shield that projects unsafe actions to safe ones.
Supports condition parsing and action effects (set, limit, rate_limit).

Usage:
    from hai_ml.safety.shield import Shield
    shield = Shield(schema_path='hai_ml/schemas/p3.yaml')
    safe_action, info = shield.project(state_dict, action_vector)
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from ruamel.yaml import YAML


@dataclass
class SafetyRule:
    """Parsed safety rule with condition and effects."""
    id: str
    description: str
    condition_str: str
    action_str: str
    severity: str
    condition_fn: Optional[Callable[[Dict[str, float], Dict[str, float]], bool]] = None
    effects: List[Tuple[str, Any]] = field(default_factory=list)


class RuleParser:
    """
    Parser for safety rule conditions and actions.
    
    Supports conditions like:
        - "P3_LIT01 > 90.0"
        - "P3_LIT01 < 15.0"
        - "P3_FIT02 > P3_FIT01 * 1.1"
        - "abs(delta_P3_MV01) > 10.0"
        - "P1_LIT01 < 10.0 and P1_PIT01 > 16.0"
        - "P1_LIT01 < 10.0 or P1_PIT01 > 16.0"
    
    Supports actions like:
        - "set P3_MV01 := 80.0"
        - "limit P3_MV01 in [0.0, 50.0]"
        - "rate_limit P3_MV01 to 10.0"
        - Multiple actions separated by ";"
    """
    
    # Regex patterns
    COMPARISON_OPS = {
        '>': lambda a, b: a > b,
        '<': lambda a, b: a < b,
        '>=': lambda a, b: a >= b,
        '<=': lambda a, b: a <= b,
        '==': lambda a, b: abs(a - b) < 1e-6,
        '!=': lambda a, b: abs(a - b) >= 1e-6,
    }
    
    def __init__(self, action_names: List[str]):
        self.action_names = action_names
    
    def parse_condition(self, cond_str: str) -> Callable[[Dict[str, float], Dict[str, float]], bool]:
        """
        Parse condition string into a callable function.
        
        Args:
            cond_str: Condition expression
            
        Returns:
            Function(state_dict, delta_dict) -> bool
        """
        # Handle 'and' / 'or' combinations
        if ' and ' in cond_str:
            parts = cond_str.split(' and ')
            fns = [self.parse_condition(p.strip()) for p in parts]
            return lambda s, d: all(fn(s, d) for fn in fns)
        
        if ' or ' in cond_str:
            parts = cond_str.split(' or ')
            fns = [self.parse_condition(p.strip()) for p in parts]
            return lambda s, d: any(fn(s, d) for fn in fns)
        
        # Parse single condition
        return self._parse_single_condition(cond_str.strip())
    
    def _parse_single_condition(self, cond_str: str) -> Callable[[Dict[str, float], Dict[str, float]], bool]:
        """Parse a single comparison condition."""
        # Handle abs(delta_X) pattern
        abs_match = re.match(r'abs\(delta_(\w+)\)\s*(>|<|>=|<=|==|!=)\s*([0-9.]+)', cond_str)
        if abs_match:
            tag = abs_match.group(1)
            op = abs_match.group(2)
            val = float(abs_match.group(3))
            op_fn = self.COMPARISON_OPS[op]
            return lambda s, d: op_fn(abs(d.get(tag, 0.0)), val)
        
        # Handle tag1 op tag2 * factor pattern
        cross_match = re.match(r'(\w+)\s*(>|<|>=|<=)\s*(\w+)\s*\*\s*([0-9.]+)', cond_str)
        if cross_match:
            tag1 = cross_match.group(1)
            op = cross_match.group(2)
            tag2 = cross_match.group(3)
            factor = float(cross_match.group(4))
            op_fn = self.COMPARISON_OPS[op]
            return lambda s, d: op_fn(s.get(tag1, 0.0), s.get(tag2, 0.0) * factor)
        
        # Handle tag1 op tag2 +/- value pattern
        cross_add_match = re.match(r'(\w+)\s*(>|<|>=|<=)\s*(\w+)\s*([+-])\s*([0-9.]+)', cond_str)
        if cross_add_match:
            tag1 = cross_add_match.group(1)
            op = cross_add_match.group(2)
            tag2 = cross_add_match.group(3)
            sign = 1 if cross_add_match.group(4) == '+' else -1
            offset = float(cross_add_match.group(5))
            op_fn = self.COMPARISON_OPS[op]
            return lambda s, d: op_fn(s.get(tag1, 0.0), s.get(tag2, 0.0) + sign * offset)
        
        # Handle simple tag op value pattern
        simple_match = re.match(r'(\w+)\s*(>|<|>=|<=|==|!=)\s*([0-9.]+)', cond_str)
        if simple_match:
            tag = simple_match.group(1)
            op = simple_match.group(2)
            val = float(simple_match.group(3))
            op_fn = self.COMPARISON_OPS[op]
            return lambda s, d: op_fn(s.get(tag, 0.0), val)
        
        # Default: always false
        print(f"Warning: Could not parse condition: {cond_str}")
        return lambda s, d: False
    
    def parse_action(self, action_str: str) -> List[Tuple[str, Any]]:
        """
        Parse action string into a list of effects.
        
        Returns list of tuples:
            ('set', tag, value)
            ('limit', tag, low, high)
            ('rate_limit', tag, max_rate)
        """
        effects = []
        
        # Split by semicolon for multiple actions
        parts = [p.strip() for p in action_str.split(';') if p.strip()]
        
        for part in parts:
            # Parse "set TAG := VALUE"
            set_match = re.match(r'set\s+(\w+)\s*:=\s*([0-9.]+)', part)
            if set_match:
                tag = set_match.group(1)
                val = float(set_match.group(2))
                effects.append(('set', tag, val))
                continue
            
            # Parse "limit TAG in [LOW, HIGH]"
            limit_match = re.match(r'limit\s+(\w+)\s+in\s*\[([0-9.]+)\s*,\s*([0-9.]+)\]', part)
            if limit_match:
                tag = limit_match.group(1)
                low = float(limit_match.group(2))
                high = float(limit_match.group(3))
                effects.append(('limit', tag, low, high))
                continue
            
            # Parse "rate_limit TAG to VALUE"
            rate_match = re.match(r'rate_limit\s+(\w+)\s+to\s+([0-9.]+)', part)
            if rate_match:
                tag = rate_match.group(1)
                max_rate = float(rate_match.group(2))
                effects.append(('rate_limit', tag, max_rate))
                continue
            
            print(f"Warning: Could not parse action: {part}")
        
        return effects


class Shield:
    """
    Safety shield that projects actions to safe region.
    
    Loads rules from YAML schema and applies them to prevent
    unsafe states/actions.
    """
    
    def __init__(
        self,
        schema_path: Optional[str] = None,
        schema: Optional[dict] = None,
        action_names: Optional[List[str]] = None,
        dt: float = 1.0,
    ):
        """
        Initialize shield from schema.
        
        Args:
            schema_path: Path to YAML schema file
            schema: Pre-loaded schema dict (alternative to path)
            action_names: List of action tag names
            dt: Time step for rate limiting
        """
        self.dt = dt
        
        # Load schema
        if schema is not None:
            self.schema = schema
        elif schema_path is not None:
            yaml = YAML(typ='safe')
            with open(schema_path, 'r') as f:
                self.schema = yaml.load(f)
        else:
            raise ValueError("Must provide schema_path or schema")
        
        # Extract action configuration
        if action_names is not None:
            self.action_names = action_names
        else:
            self.action_names = [tag['name'] for tag in self.schema['action_tags']]
        
        self.n_actions = len(self.action_names)
        self.action_name_to_idx = {name: i for i, name in enumerate(self.action_names)}
        
        # Get action bounds
        self.action_bounds = {}
        self.action_rate_limits = {}
        for tag in self.schema['action_tags']:
            name = tag['name']
            self.action_bounds[name] = (tag.get('min', 0.0), tag.get('max', 100.0))
            self.action_rate_limits[name] = tag.get('rate_limit', float('inf'))
        
        # Parse rules
        self.parser = RuleParser(self.action_names)
        self.rules: List[SafetyRule] = []
        self._parse_rules()
        
        # Track previous action for rate limiting
        self._prev_action: Optional[np.ndarray] = None
        
        # Statistics
        self.total_interventions = 0
        self.rule_hits: Dict[str, int] = {rule.id: 0 for rule in self.rules}
    
    def _parse_rules(self):
        """Parse all safety rules from schema."""
        safety_rules = self.schema.get('safety_rules', [])
        
        for rule_def in safety_rules:
            rule = SafetyRule(
                id=rule_def['id'],
                description=rule_def['description'],
                condition_str=rule_def['condition'],
                action_str=rule_def['action'],
                severity=rule_def.get('severity', 'medium'),
            )
            
            # Parse condition
            try:
                rule.condition_fn = self.parser.parse_condition(rule.condition_str)
            except Exception as e:
                print(f"Warning: Failed to parse condition for {rule.id}: {e}")
                rule.condition_fn = lambda s, d: False
            
            # Parse effects
            try:
                rule.effects = self.parser.parse_action(rule.action_str)
            except Exception as e:
                print(f"Warning: Failed to parse action for {rule.id}: {e}")
                rule.effects = []
            
            self.rules.append(rule)
        
        print(f"Loaded {len(self.rules)} safety rules")
    
    def reset(self):
        """Reset shield state (call at episode start)."""
        self._prev_action = None
        self.total_interventions = 0
        self.rule_hits = {rule.id: 0 for rule in self.rules}
    
    def project(
        self,
        state: Dict[str, float],
        action: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Project action to safe region.
        
        Args:
            state: Dictionary mapping state tag names to values
            action: Action vector (same order as action_names)
            
        Returns:
            (safe_action, info_dict)
        """
        # Start with original action
        safe_action = action.copy()
        
        # Compute deltas for rate limit rules
        if self._prev_action is not None:
            deltas = {
                name: float(action[i] - self._prev_action[i])
                for i, name in enumerate(self.action_names)
            }
        else:
            deltas = {name: 0.0 for name in self.action_names}
        
        # Track interventions
        intervened = False
        triggered_rules = []
        
        # Check each rule
        for rule in self.rules:
            if rule.condition_fn is not None:
                try:
                    if rule.condition_fn(state, deltas):
                        # Rule triggered - apply effects
                        triggered_rules.append(rule.id)
                        self.rule_hits[rule.id] += 1
                        
                        for effect in rule.effects:
                            action_change = self._apply_effect(safe_action, effect)
                            if action_change:
                                intervened = True
                except Exception as e:
                    # Don't let rule errors crash the system
                    pass
        
        # Apply global action bounds
        for i, name in enumerate(self.action_names):
            low, high = self.action_bounds.get(name, (0.0, 100.0))
            if safe_action[i] < low or safe_action[i] > high:
                safe_action[i] = np.clip(safe_action[i], low, high)
                intervened = True
        
        # Apply global rate limits
        if self._prev_action is not None:
            for i, name in enumerate(self.action_names):
                max_rate = self.action_rate_limits.get(name, float('inf'))
                delta = safe_action[i] - self._prev_action[i]
                max_delta = max_rate * self.dt
                if abs(delta) > max_delta:
                    safe_action[i] = self._prev_action[i] + np.sign(delta) * max_delta
                    intervened = True
        
        # Update state
        self._prev_action = safe_action.copy()
        if intervened:
            self.total_interventions += 1
        
        info = {
            'intervened': intervened,
            'triggered_rules': triggered_rules,
            'original_action': action.tolist(),
            'safe_action': safe_action.tolist(),
        }
        
        return safe_action, info
    
    def _apply_effect(self, action: np.ndarray, effect: Tuple) -> bool:
        """
        Apply a single effect to the action vector.
        
        Returns True if action was modified.
        """
        effect_type = effect[0]
        tag = effect[1]
        
        if tag not in self.action_name_to_idx:
            return False
        
        idx = self.action_name_to_idx[tag]
        original = action[idx]
        
        if effect_type == 'set':
            value = effect[2]
            action[idx] = value
        
        elif effect_type == 'limit':
            low, high = effect[2], effect[3]
            action[idx] = np.clip(action[idx], low, high)
        
        elif effect_type == 'rate_limit':
            max_rate = effect[2]
            if self._prev_action is not None:
                delta = action[idx] - self._prev_action[idx]
                max_delta = max_rate * self.dt
                if abs(delta) > max_delta:
                    action[idx] = self._prev_action[idx] + np.sign(delta) * max_delta
        
        return abs(action[idx] - original) > 1e-6
    
    def get_intervention_rate(self, total_steps: int) -> float:
        """Get intervention rate over given number of steps."""
        if total_steps == 0:
            return 0.0
        return self.total_interventions / total_steps
    
    def get_rule_statistics(self) -> Dict[str, int]:
        """Get hit counts for each rule."""
        return self.rule_hits.copy()


def create_shield(task: str, **kwargs) -> Shield:
    """
    Factory function to create shield for a task.
    
    Args:
        task: Task name ('p1', 'p3', 'p12')
        **kwargs: Additional arguments for Shield
        
    Returns:
        Shield instance
    """
    schema_paths = {
        'p1': 'hai_ml/schemas/p1.yaml',
        'p3': 'hai_ml/schemas/p3.yaml',
        'p12': 'hai_ml/schemas/p12.yaml',
    }
    
    if task not in schema_paths:
        raise ValueError(f"Unknown task: {task}")
    
    # Try to find schema relative to this file
    base_path = Path(__file__).parent.parent
    full_path = base_path / 'schemas' / f'{task}.yaml'
    
    if full_path.exists():
        schema_path = str(full_path)
    else:
        schema_path = schema_paths[task]
    
    return Shield(schema_path=schema_path, **kwargs)
