"""Train the goal-conditioned SAC reacher that executes options (any start to any goal inside the 10 cm ring).

    python train_reacher.py --seed 7 --out runs/reacher

Observation (22): goal xy, fingertip xy, normalised muscle lengths and velocities, activations. Every 10k steps it
reaches 100 fixed start/goal pairs; best_model.zip keeps the best (success, -path ratio, -deviation, -duration).
Training stops after 3 evaluations in a row with success >= 0.99, path ratio <= 1.2 and deviation <= 2 cm.
"""
import argparse
import json
from pathlib import Path

import gymnasium as gym
import motornet as mn
import numpy as np
import torch
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import arm
from train_agent import line_deviation

WORKSPACE, SEPARATION, HIT_RADIUS, MAX_STEPS = 0.1, 0.02, 0.02, 200
EVALUATE_EVERY, PAIRS, PAIR_SEED = 10_000, 100, 10007


def sample_disk(rng, center, radius=WORKSPACE):
    distance, angle = radius * np.sqrt(rng.random()), 2 * np.pi * rng.random()
    return (np.asarray(center, np.float64) + np.stack((distance * np.cos(angle), distance * np.sin(angle)))).astype(np.float32)


def sample_pair(rng, center):
    start = sample_disk(rng, center)
    while True:
        goal = sample_disk(rng, center)
        if np.linalg.norm(start - goal) >= SEPARATION:
            return start, goal


class ReacherTask(mn.environment.Environment):
    reference_posture = np.deg2rad([45., 90.]).astype(np.float32)

    def __init__(self, effector):
        super().__init__(effector=effector, name='ReacherTask', differentiable=False, max_ep_duration=MAX_STEPS * effector.dt,
                         action_noise=0., obs_noise=0., proprioception_delay=0., vision_delay=0.)
        reference = torch.tensor([[*self.reference_posture, 0., 0.]], dtype=torch.float32)
        self.register_buffer('workspace_center', self.joint2cartesian(reference)[0, :2])

    def _get_obs_size(self):
        return 22

    def get_obs(self, action=None, deterministic=True):
        muscle = self.states['muscle']
        lengths = (muscle[:, 1:2, :] / self.muscle.l0_ce).squeeze(1)
        velocities = (muscle[:, 2:3, :] / self.muscle.vmax).squeeze(1)
        return self.detach(torch.cat((self.subgoal, self.states['fingertip'], lengths, velocities, muscle[:, 0, :]), dim=-1))

    def joints(self, xy):
        l1, l2 = self.skeleton.L1, self.skeleton.L2
        xy = np.asarray(xy, dtype=np.float64)
        elbow = np.arccos(np.clip((xy @ xy - l1 ** 2 - l2 ** 2) / (2 * l1 * l2), -1., 1.))
        shoulder = np.arctan2(xy[1], xy[0]) - np.arctan2(l2 * np.sin(elbow), l1 + l2 * np.cos(elbow))
        return np.asarray((shoulder, elbow), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self._set_generator(seed=seed)
        options = options or {}
        center = self.workspace_center.detach().cpu().numpy()
        if 'start_xy' in options:
            start, goal = np.asarray(options['start_xy'], np.float32), np.asarray(options['target_xy'], np.float32)
        else:
            start, goal = sample_pair(self.np_random, center)
        joint_state = torch.tensor([[*self.joints(start), 0., 0.]], dtype=torch.float32)
        self.subgoal = torch.tensor(goal[None], dtype=torch.float32)
        self.finished = torch.zeros(1, dtype=torch.bool)
        self.elapsed_steps = 0
        super().reset(seed=None, options={'batch_size': 1, 'joint_state': joint_state, 'deterministic': True})
        self.goal = self.subgoal
        return self.get_obs(), {'states': self._maybe_detach_states(), 'finished': self.detach(self.finished)}

    def step(self, action, deterministic=True, **kwargs):
        action = torch.as_tensor(action, dtype=torch.float32)
        self.elapsed_steps += 1
        self.elapsed = self.elapsed_steps * self.dt
        before = self.states['fingertip'].clone()
        before_distance = torch.linalg.vector_norm(before - self.subgoal, dim=1)
        self.effector.step(action, **kwargs)
        after = self.states['fingertip'].clone()
        after_distance = torch.linalg.vector_norm(after - self.subgoal, dim=1)
        progress = (before_distance - after_distance) / 0.01
        step_distance = torch.linalg.vector_norm(after - before, dim=1)
        wasted = torch.clamp(step_distance - (before_distance - after_distance), min=0.)
        hit = (after_distance <= HIT_RADIUS) & ~self.finished
        self.finished |= hit
        reward = progress - 0.1 * after_distance / 0.10 - 0.05 * wasted / 0.01 + 20. * hit.float() + 200. * hit.float()
        terminated = bool(torch.all(self.finished).item())
        truncated = self.elapsed_steps >= MAX_STEPS and not terminated
        info = {'states': self._maybe_detach_states(), 'finished': self.detach(self.finished),
                'step_distance': self.detach(step_distance), 'distance_after': self.detach(after_distance)}
        return self.get_obs(), self.detach(reward[:, None]), terminated, truncated, info


class ReacherGym(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self):
        self.task = ReacherTask(mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle(), timestep=.01))
        self.observation_space = self.task.observation_space
        self.action_space = gym.spaces.Box(-1., 1., (6,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        observation, info = self.task.reset(seed=seed, options=options)
        info = arm.squeeze(info)
        info['is_success'] = bool(info['finished'])
        return np.asarray(observation[0], dtype=np.float32), info

    def step(self, action):
        excitation = (np.clip(np.asarray(action, dtype=np.float32), -1., 1.) + 1.) / 2.
        observation, reward, terminated, truncated, info = self.task.step(excitation[None, :])
        info = arm.squeeze(info)
        info['is_success'] = bool(info['finished'])
        return np.asarray(observation[0], dtype=np.float32), float(reward[0, 0]), bool(terminated), bool(truncated), info


def evaluate(model, env, pairs):
    results = []
    for start, goal in pairs:
        obs, info = env.reset(options={'start_xy': start, 'target_xy': goal})
        path = deviation = 0.
        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            path += float(info['step_distance'])
            deviation = max(deviation, line_deviation(np.asarray(info['states']['fingertip'], np.float32), start, goal))
            if terminated or truncated:
                break
        results.append((info['is_success'], path / max(float(np.linalg.norm(goal - start)), 1e-8), deviation, step + 1))
    ok = [r for r in results if r[0]]
    success = float(np.mean([r[0] for r in results]))
    ratio = float(np.mean([r[1] for r in ok])) if ok else np.inf
    duration = float(np.mean([r[3] for r in ok])) if ok else np.inf
    return success, ratio, float(max(r[2] for r in results)), duration


class Evaluation(BaseCallback):
    def __init__(self, out, pairs, progress, bar):
        super().__init__()
        self.out, self.pairs, self.progress, self.bar = out, pairs, progress, bar
        self.best, self.streak, self.history = None, 0, []

    def _init_callback(self):
        self.env = ReacherGym()

    def _evaluate(self):
        success, ratio, deviation, duration = evaluate(self.model, self.env, self.pairs)
        self.history.append(dict(step=self.num_timesteps, success=success, path_ratio=ratio, deviation=deviation))
        (self.out / 'evaluations.json').write_text(json.dumps(self.history, indent=1))
        key = (success, -ratio, -deviation, -duration)
        if self.best is None or key > self.best:
            self.best = key
            self.model.save(str(self.out / 'best_model'))
        self.streak = self.streak + 1 if success >= .99 and ratio <= 1.2 and deviation <= .02 else 0
        self.progress.update(self.bar, description=f'reacher (eval success {success:.2f})')

    def _on_step(self):
        self.progress.update(self.bar, completed=self.num_timesteps)
        if self.n_calls % EVALUATE_EVERY == 0:
            self._evaluate()
            return self.streak < 3
        return True

    def _on_training_end(self):
        self._evaluate()


def train(seed, steps, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(PAIR_SEED)
    center = ReacherGym().task.workspace_center.detach().cpu().numpy()
    pairs = [sample_pair(rng, center) for _ in range(PAIRS)]
    env = DummyVecEnv([lambda: Monitor(ReacherGym(), filename=str(out / 'worker_0'), info_keywords=('is_success',))])
    model = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=1_000_000, learning_starts=10_000, batch_size=256,
                gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1, ent_coef='auto', seed=seed, device='cpu', verbose=0)
    columns = (TextColumn('{task.description}'), BarColumn(), TextColumn('{task.completed:,.0f}/{task.total:,.0f}'),
               TimeElapsedColumn(), TimeRemainingColumn())
    with Progress(*columns) as progress:
        bar = progress.add_task('reacher', total=steps)
        model.learn(total_timesteps=steps, callback=Evaluation(out, pairs, progress, bar))
    model.save(str(out / 'last_model'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    train(args.seed, args.steps, args.out)
