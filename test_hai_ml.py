"""
Quick test script for hai_ml components.
"""
import numpy as np

print("=" * 60)
print("HAI-ML Component Tests")
print("=" * 60)

# Test 1: Environment
print("\n[1] Testing HaiEnv environment...")
from hai_ml.envs.hai_gym import HaiEnv

env = HaiEnv(schema_path='hai_ml/schemas/p3.yaml', max_steps=100)
print(f"    Observation space: {env.observation_space}")
print(f"    Action space: {env.action_space}")

obs, info = env.reset()
print(f"    Initial observation shape: {obs.shape}")

total_reward = 0
for i in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    
print(f"    Ran 10 steps, total reward: {total_reward:.2f}")
print("    ✓ Environment works!")

# Test 2: Shield
print("\n[2] Testing Shield...")
from hai_ml.safety.shield import Shield

shield = Shield(schema_path='hai_ml/schemas/p3.yaml')
print(f"    Loaded {len(shield.rules)} safety rules")

# Create state and action as numpy arrays
state = np.array([900.0, 2.0, 30.0, 7.0, 300.0, 800.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
action = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
safe_action, proj_info = shield.project(state, action)
print(f"    Tested projection: intervened={proj_info['intervened']}")
print("    ✓ Shield works!")

# Test 3: Environment with different schema
print("\n[3] Testing HaiEnv with P1 schema...")
env_p1 = HaiEnv(schema_path='hai_ml/schemas/p1.yaml')
obs, info = env_p1.reset()
print(f"    P1 observation shape: {obs.shape}")
print(f"    P1 action space: {env_p1.action_space}")
for i in range(10):
    action = env_p1.action_space.sample()
    obs, reward, done, truncated, info = env_p1.step(action)
print("    ✓ P1 environment works!")

# Test 4: Environment with P12 schema
print("\n[4] Testing HaiEnv with P12 coupled schema...")
env_p12 = HaiEnv(schema_path='hai_ml/schemas/p12.yaml')
obs, info = env_p12.reset()
print(f"    P12 observation shape: {obs.shape}")
print(f"    P12 action space: {env_p12.action_space}")
for i in range(10):
    action = env_p12.action_space.sample()
    obs, reward, done, truncated, info = env_p12.step(action)
print("    ✓ P12 environment works!")

# Test 5: Metrics
print("\n[5] Testing metrics module...")
from hai_ml.eval.metrics import itae, ise, wear

# Generate some fake trajectory data
setpoints = np.ones(100) * 50
trajectory = 50 + np.random.randn(100) * 5
errors = trajectory - setpoints
actions = np.random.randn(100, 4)

itae_val = itae(errors)
ise_val = ise(errors)
wear_val = wear(actions)

print(f"    ITAE: {itae_val:.2f}")
print(f"    ISE: {ise_val:.2f}")
print(f"    Wear: {wear_val:.2f}")
print("    ✓ Metrics work!")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
