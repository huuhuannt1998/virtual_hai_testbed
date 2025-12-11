"""
Unit Tests for Safety Shield
============================

Tests shield projection behavior on various rules.

Run with:
    python -m pytest hai_ml/safety/tests/test_shield.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from hai_ml.safety.shield import Shield, RuleParser, SafetyRule


# Sample schema for testing
TEST_SCHEMA = {
    'process': 'test',
    'action_tags': [
        {'name': 'MV01', 'min': 0.0, 'max': 100.0, 'rate_limit': 10.0},
        {'name': 'MV02', 'min': 0.0, 'max': 100.0, 'rate_limit': 5.0},
        {'name': 'PUMP_SPEED', 'min': 20.0, 'max': 100.0, 'rate_limit': 3.0},
    ],
    'state_tags': [
        {'name': 'LIT01', 'nominal_min': 10.0, 'nominal_max': 90.0},
        {'name': 'PIT01', 'nominal_min': 1.0, 'nominal_max': 10.0},
        {'name': 'FIT01', 'nominal_min': 0.0, 'nominal_max': 50.0},
    ],
    'safety_rules': [
        {
            'id': 'R01',
            'description': 'High level protection',
            'condition': 'LIT01 > 85.0',
            'action': 'limit MV01 in [0.0, 30.0]',
            'severity': 'high',
        },
        {
            'id': 'R02',
            'description': 'Low level protection',
            'condition': 'LIT01 < 15.0',
            'action': 'set MV01 := 80.0',
            'severity': 'high',
        },
        {
            'id': 'R03',
            'description': 'Overpressure protection',
            'condition': 'PIT01 > 9.0',
            'action': 'limit PUMP_SPEED in [20.0, 50.0]',
            'severity': 'critical',
        },
        {
            'id': 'R04',
            'description': 'Combined condition',
            'condition': 'LIT01 < 20.0 and PIT01 > 8.0',
            'action': 'set MV02 := 0.0; limit PUMP_SPEED in [20.0, 40.0]',
            'severity': 'critical',
        },
        {
            'id': 'R05',
            'description': 'Rate limit rule',
            'condition': 'abs(delta_MV01) > 10.0',
            'action': 'rate_limit MV01 to 10.0',
            'severity': 'low',
        },
        {
            'id': 'R06',
            'description': 'Cross-tag comparison',
            'condition': 'FIT01 > LIT01 * 0.5',
            'action': 'limit MV02 in [0.0, 50.0]',
            'severity': 'medium',
        },
    ],
}


class TestRuleParser:
    """Tests for RuleParser class."""
    
    def setup_method(self):
        self.parser = RuleParser(['MV01', 'MV02', 'PUMP_SPEED'])
    
    def test_parse_simple_greater_than(self):
        """Test parsing 'TAG > value' condition."""
        fn = self.parser.parse_condition('LIT01 > 85.0')
        assert fn({'LIT01': 90.0}, {}) == True
        assert fn({'LIT01': 80.0}, {}) == False
        assert fn({'LIT01': 85.0}, {}) == False
    
    def test_parse_simple_less_than(self):
        """Test parsing 'TAG < value' condition."""
        fn = self.parser.parse_condition('LIT01 < 15.0')
        assert fn({'LIT01': 10.0}, {}) == True
        assert fn({'LIT01': 20.0}, {}) == False
    
    def test_parse_and_condition(self):
        """Test parsing 'A and B' condition."""
        fn = self.parser.parse_condition('LIT01 < 20.0 and PIT01 > 8.0')
        assert fn({'LIT01': 15.0, 'PIT01': 9.0}, {}) == True
        assert fn({'LIT01': 25.0, 'PIT01': 9.0}, {}) == False
        assert fn({'LIT01': 15.0, 'PIT01': 7.0}, {}) == False
    
    def test_parse_or_condition(self):
        """Test parsing 'A or B' condition."""
        fn = self.parser.parse_condition('LIT01 < 20.0 or PIT01 > 8.0')
        assert fn({'LIT01': 15.0, 'PIT01': 5.0}, {}) == True
        assert fn({'LIT01': 25.0, 'PIT01': 9.0}, {}) == True
        assert fn({'LIT01': 25.0, 'PIT01': 7.0}, {}) == False
    
    def test_parse_abs_delta(self):
        """Test parsing 'abs(delta_TAG) > value' condition."""
        fn = self.parser.parse_condition('abs(delta_MV01) > 10.0')
        assert fn({}, {'MV01': 15.0}) == True
        assert fn({}, {'MV01': -15.0}) == True
        assert fn({}, {'MV01': 5.0}) == False
    
    def test_parse_cross_tag_multiply(self):
        """Test parsing 'TAG1 > TAG2 * factor' condition."""
        fn = self.parser.parse_condition('FIT01 > LIT01 * 0.5')
        assert fn({'FIT01': 30.0, 'LIT01': 40.0}, {}) == True  # 30 > 20
        assert fn({'FIT01': 15.0, 'LIT01': 40.0}, {}) == False  # 15 < 20
    
    def test_parse_set_action(self):
        """Test parsing 'set TAG := value' action."""
        effects = self.parser.parse_action('set MV01 := 80.0')
        assert len(effects) == 1
        assert effects[0] == ('set', 'MV01', 80.0)
    
    def test_parse_limit_action(self):
        """Test parsing 'limit TAG in [low, high]' action."""
        effects = self.parser.parse_action('limit MV01 in [0.0, 30.0]')
        assert len(effects) == 1
        assert effects[0] == ('limit', 'MV01', 0.0, 30.0)
    
    def test_parse_rate_limit_action(self):
        """Test parsing 'rate_limit TAG to value' action."""
        effects = self.parser.parse_action('rate_limit MV01 to 10.0')
        assert len(effects) == 1
        assert effects[0] == ('rate_limit', 'MV01', 10.0)
    
    def test_parse_multi_action(self):
        """Test parsing multiple actions separated by ';'."""
        effects = self.parser.parse_action('set MV02 := 0.0; limit PUMP_SPEED in [20.0, 40.0]')
        assert len(effects) == 2
        assert effects[0] == ('set', 'MV02', 0.0)
        assert effects[1] == ('limit', 'PUMP_SPEED', 20.0, 40.0)


class TestShield:
    """Tests for Shield class."""
    
    def setup_method(self):
        self.shield = Shield(schema=TEST_SCHEMA, dt=1.0)
    
    def test_no_intervention_normal_state(self):
        """Test that normal states don't trigger intervention."""
        state = {'LIT01': 50.0, 'PIT01': 5.0, 'FIT01': 20.0}
        action = np.array([50.0, 50.0, 60.0])
        
        safe_action, info = self.shield.project(state, action)
        
        # Should be unchanged
        np.testing.assert_array_almost_equal(safe_action, action)
        assert info['intervened'] == False
    
    def test_high_level_triggers_limit(self):
        """Test R01: High level limits MV01."""
        self.shield.reset()
        state = {'LIT01': 90.0, 'PIT01': 5.0, 'FIT01': 20.0}
        action = np.array([80.0, 50.0, 60.0])  # MV01 = 80, should be limited to 30
        
        safe_action, info = self.shield.project(state, action)
        
        assert safe_action[0] <= 30.0
        assert info['intervened'] == True
        assert 'R01' in info['triggered_rules']
    
    def test_low_level_triggers_set(self):
        """Test R02: Low level sets MV01 to 80."""
        self.shield.reset()
        state = {'LIT01': 10.0, 'PIT01': 5.0, 'FIT01': 20.0}
        action = np.array([20.0, 50.0, 60.0])  # MV01 = 20, should be set to 80
        
        safe_action, info = self.shield.project(state, action)
        
        assert safe_action[0] == 80.0
        assert info['intervened'] == True
        assert 'R02' in info['triggered_rules']
    
    def test_overpressure_limits_pump(self):
        """Test R03: Overpressure limits PUMP_SPEED."""
        self.shield.reset()
        state = {'LIT01': 50.0, 'PIT01': 9.5, 'FIT01': 20.0}
        action = np.array([50.0, 50.0, 80.0])  # PUMP_SPEED = 80, should be limited to 50
        
        safe_action, info = self.shield.project(state, action)
        
        assert safe_action[2] <= 50.0
        assert info['intervened'] == True
        assert 'R03' in info['triggered_rules']
    
    def test_combined_condition_triggers(self):
        """Test R04: Combined condition triggers multiple effects."""
        self.shield.reset()
        state = {'LIT01': 15.0, 'PIT01': 8.5, 'FIT01': 20.0}
        action = np.array([50.0, 70.0, 80.0])
        
        safe_action, info = self.shield.project(state, action)
        
        assert safe_action[1] == 0.0  # MV02 set to 0
        assert safe_action[2] <= 40.0  # PUMP_SPEED limited
        assert 'R04' in info['triggered_rules']
    
    def test_global_bounds_enforced(self):
        """Test that global action bounds are always enforced."""
        self.shield.reset()
        state = {'LIT01': 50.0, 'PIT01': 5.0, 'FIT01': 20.0}
        action = np.array([150.0, -10.0, 10.0])  # Out of bounds
        
        safe_action, info = self.shield.project(state, action)
        
        assert safe_action[0] == 100.0  # Clipped to max
        assert safe_action[1] == 0.0    # Clipped to min
        assert safe_action[2] == 20.0   # Clipped to min (20)
    
    def test_rate_limiting(self):
        """Test that rate limits are enforced."""
        self.shield.reset()
        state = {'LIT01': 50.0, 'PIT01': 5.0, 'FIT01': 20.0}
        
        # First action establishes baseline
        action1 = np.array([50.0, 50.0, 50.0])
        self.shield.project(state, action1)
        
        # Second action tries to change too fast
        action2 = np.array([80.0, 50.0, 50.0])  # +30 on MV01, rate limit is 10
        safe_action, info = self.shield.project(state, action2)
        
        # Should only increase by rate limit
        assert safe_action[0] <= 60.0  # 50 + 10
    
    def test_intervention_counting(self):
        """Test that interventions are counted correctly."""
        self.shield.reset()
        
        # Trigger some interventions
        state_high = {'LIT01': 90.0, 'PIT01': 5.0, 'FIT01': 20.0}
        action = np.array([80.0, 50.0, 60.0])
        
        for _ in range(5):
            self.shield.project(state_high, action)
        
        rate = self.shield.get_intervention_rate(5)
        assert rate == 1.0  # All steps had interventions
    
    def test_rule_statistics(self):
        """Test that rule hit statistics are tracked."""
        self.shield.reset()
        
        # Trigger R01 multiple times
        state_high = {'LIT01': 90.0, 'PIT01': 5.0, 'FIT01': 20.0}
        action = np.array([80.0, 50.0, 60.0])
        
        for _ in range(3):
            self.shield.project(state_high, action)
        
        stats = self.shield.get_rule_statistics()
        assert stats['R01'] == 3


class TestDeterminism:
    """Tests for deterministic behavior."""
    
    def test_same_input_same_output(self):
        """Test that same inputs produce same outputs."""
        shield1 = Shield(schema=TEST_SCHEMA, dt=1.0)
        shield2 = Shield(schema=TEST_SCHEMA, dt=1.0)
        
        state = {'LIT01': 90.0, 'PIT01': 5.0, 'FIT01': 20.0}
        action = np.array([80.0, 50.0, 60.0])
        
        safe1, _ = shield1.project(state, action)
        safe2, _ = shield2.project(state, action)
        
        np.testing.assert_array_equal(safe1, safe2)
    
    def test_sequence_determinism(self):
        """Test that sequence of actions produces deterministic results."""
        shield1 = Shield(schema=TEST_SCHEMA, dt=1.0)
        shield2 = Shield(schema=TEST_SCHEMA, dt=1.0)
        
        states = [
            {'LIT01': 50.0, 'PIT01': 5.0, 'FIT01': 20.0},
            {'LIT01': 60.0, 'PIT01': 6.0, 'FIT01': 25.0},
            {'LIT01': 90.0, 'PIT01': 9.5, 'FIT01': 30.0},
        ]
        actions = [
            np.array([50.0, 50.0, 50.0]),
            np.array([60.0, 60.0, 60.0]),
            np.array([80.0, 80.0, 80.0]),
        ]
        
        results1 = []
        results2 = []
        
        for state, action in zip(states, actions):
            safe1, _ = shield1.project(state, action)
            results1.append(safe1.copy())
        
        for state, action in zip(states, actions):
            safe2, _ = shield2.project(state, action)
            results2.append(safe2.copy())
        
        for r1, r2 in zip(results1, results2):
            np.testing.assert_array_equal(r1, r2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
