# Robustness Evaluation Pipeline Forensic Analysis

## TASK 1 — Trace the evaluation pipeline
The evaluation pipeline in `evaluate_robustness.py` executes linearly:
1. **Model Loading:** The checkpoint is loaded exactly once using `load_agents(args.checkpoint)` and wrapped in `LearnedPolicy`.
2. **Environment Creation:** `MultiUAVEnv(env_cfg)` is initialized once with default parameters from `config/default.yaml`.
3. **Experiment Loop:** The script loops through experiments sequentially: `sensor_noise` → `sensor_range` → `goal_distribution` → `variable_speed`.
4. **Scenario Overrides:** For each sweep, a temporary `stage_cfg` is built by merging the scenario's base config (e.g., `S1_Static_Dynamic`) with the sweep's override parameters.
5. **Environment Reset:** `env.reset(stage_cfg)` is called, which updates environment dimensions, goal distances, and sensor configurations.
6. **Observation Generation:** `_get_single_observation()` calls `Rangefinder.scan()` to generate LiDAR observations and noise.
7. **Evaluation Loop:** `policy.act()` computes actions, `env.step()` advances physics, and metrics (Success Rate, Collision Rate, Time to Goal) are aggregated.

## TASK 2 — Verify checkpoint consistency
**Result:** Safe. The checkpoint is loaded exactly **once** before the experiment loops begin:
```python
# evaluate_robustness.py:254
agents, _ = load_agents(args.checkpoint, args.config, args.device, variant=args.variant)
policy = LearnedPolicy(agents, name="MARDPG")
```
There is no accidental checkpoint switching, no dynamic loading of "best/latest" weights, and no reinitialization. Every experiment evaluates `checkpoints/cl_mardpg_seed3/interrupted_episode_4029`.

## TASK 3 — Parameter isolation

| Experiment | Changed Parameters | Unchanged Parameters | Unexpected Side Effects |
| :--- | :--- | :--- | :--- |
| **Sensor Noise** | `lidar_noise` | `sensor_range` (10m), `min_sep`, `variable_speed` | Gaussian noise clipping creates "phantom obstacles" in empty space. |
| **Sensor Range** | `sensor_range` | `lidar_noise` (0.02), `min_sep`, `variable_speed` | 1. Neural net observation normalization shifts drastically.<br>2. `RewardFunction` uses stale `range_max` (10m).<br>3. **Pollutes environment state for all subsequent experiments.** |
| **Goal Dist.** | `min_sep` | `lidar_noise` (0.02), `variable_speed` | Inherits corrupted `sensor_range=25m` from the previous experiment. |
| **Variable Speed** | `variable_speed` | `lidar_noise` (0.02), `min_sep` | Inherits corrupted `sensor_range=25m` from the previous experiment. |

## TASK 4 — Investigate Sensor Range
Increasing the sensor range severely degrades performance due to two distinct implementation bugs:

1. **Observation Normalization Shift (`sensors.py:128`)**
   ```python
   noisy_norm = np.clip(distances / self.range_max + noise, 0.0, 1.0)
   ```
   The neural network was trained with `range_max = 10.0m`. If an obstacle was 2.0m away, the network received `2.0 / 10.0 = 0.2`. When evaluated at `range_max = 25.0m`, an obstacle at 5.0m away yields `5.0 / 25.0 = 0.2`. The neural network interprets this as an obstacle 2.0m away! The agent constantly overreacts to distant obstacles, thinking a collision is imminent.

2. **Stale Reward Parameter (`rewards.py:73`)**
   ```python
   r_free = self.r_free if rangefinder_raw.flatten()[12] >= self.range_max else 0.0
   ```
   `RewardFunction` is instantiated in `MultiUAVEnv.__init__` with the default `range_max=10.0`. When `env.reset()` updates `self.rangefinder.range_max = 25.0`, it **never** updates `self.reward_fn.range_max`. Thus, the free-space reward continues using a 10.0m threshold, causing the agent to receive free-space rewards even when obstacles are 15m directly ahead.

## TASK 5 — Investigate Sensor Noise
Small noise improves performance due to an **asymmetric noise clipping bug** acting as a phantom regularization.

Code path:
```python
# sensors.py:128
noisy_norm = np.clip(distances / self.range_max
                     + self.rng.normal(0.0, sigma_l, distances.shape), 0.0, 1.0)
```
Noise is applied exactly once, but it is applied *after* dividing by `range_max` and *before* clipping to `[0.0, 1.0]`. 
In empty space, `distances / range_max == 1.0`. When Gaussian noise `N(0, sigma)` is added, we get `1.0 + N(0, sigma)`. 
When this is clipped to `[0.0, 1.0]`, **all positive noise is discarded** (clipped to 1.0), but **all negative noise is kept**. 
The expected value of empty space becomes strictly less than 1.0. The agent perceives empty space as containing soft "phantom obstacles." At small noise (σ=0.1, 0.2), these phantom obstacles induce stochastic jitter, which prevents the agent from getting stuck in symmetric local minima (APF saddle points). At high noise (σ=0.4+), the phantom obstacles appear so close that they paralyze the agent's forward progress.

## TASK 6 — Investigate baseline inconsistency
The Variable Speed baseline (74%) differs drastically from the Sensor Noise baseline (90%) because of **Environment State Pollution**.

Code path:
```python
# uav_env.py:94 (Inside MultiUAVEnv.reset())
if 'sensor_range' in stage_cfg:
    self.rangefinder.range_max = stage_cfg['sensor_range']
    self.cfg['sensor_range'] = stage_cfg['sensor_range']
```
Because the same `MultiUAVEnv` object is reused across all experiments:
1. `sensor_noise` runs first. `sensor_range` is missing from `stage_cfg`, so it uses the training default (10.0m). Performance is ~90%.
2. `sensor_range` runs second. The final sweep sets `stage_cfg['sensor_range'] = 25.0`. `self.rangefinder.range_max` is permanently mutated to 25.0m. Performance drops to 74% (due to Task 4 bugs).
3. `goal_distribution` and `variable_speed` run next. Their sweeps do NOT contain `sensor_range`. Because `env.reset()` only updates `range_max` if the key exists (`if 'sensor_range' in stage_cfg:`), it **fails to restore the default 10.0m**. 
4. Therefore, the Variable Speed baseline is evaluated entirely under a 25.0m sensor range! Its 74% success rate is an exact reflection of the polluted 25m state.

## TASK 7 — Verify scenario generation
The base scenarios (`S1_Static_Dynamic`, `S2_Longer_Distance`, `S3_Fast_Dynamic`) are generated deterministically per seed. 

However, they are slightly polluted by the `goal_distribution` sweep. In `evaluate_robustness.py`:
```python
stage_cfg.update({k: v for k, v in sweep.items() if k != 'exp_val'})
```
During the `goal_distribution` experiment, `min_sep` is overwritten to 50, 60, or 70. This correctly overrides the scenario defaults (e.g., overriding S2's default of 60.0 to 50.0). Because `stage_cfg` is rebuilt from scratch (`stage_cfg = dict(scenario_cfg)`) for every sweep, `min_sep` DOES NOT permanently pollute subsequent experiments. The scenarios are identical across experiments *except* for the polluted `sensor_range`.

## TASK 8 — Hidden implementation issues
- **Mutable configs & State Pollution**: `env.reset()` permanently mutates `self.cfg` and `self.rangefinder` instead of maintaining ephemeral episode state.
- **Missing deepcopy**: `stage_cfg` modifies `self.cfg['sensor_range']` directly.
- **Stale variables**: `RewardFunction.range_max` is initialized in `__init__` and never synchronized with `Rangefinder.range_max`.

## TASK 9 — Metrics verification
- `Success Rate`: Computed correctly via `float(reached.mean())`.
- `Collision Rate`: Computed correctly via `float(collided.mean())`.
- `Time to Goal`: Populated accurately in `evaluate_robustness.py:94` by capturing `(t+1) * dt` on the exact step `info['step_reached']` triggers.

## TASK 10 — Final verdict
1. **Are the robustness experiments scientifically valid?** **No.** The state pollution and domain-shift normalization bugs entirely invalidate the results for Sensor Range, Goal Distribution, and Variable Speed.
2. **Does each experiment isolate exactly one variable?** **No.** Every experiment after Sensor Range secretly inherits `sensor_range = 25.0m`.
3. **Hidden confounding variables?** Yes, the asymmetric noise clipping injects phantom obstacles, confounding the Sensor Noise experiment.
4. **Code-induced behaviors rather than algorithm behavior:**
   - The drop in Sensor Range performance is entirely due to the neural network misinterpreting `distances / 25.0` as extremely close objects.
   - The Variable Speed baseline drop is entirely due to state pollution.
   - The small-noise performance boost is entirely due to asymmetric negative bias preventing local minima.
