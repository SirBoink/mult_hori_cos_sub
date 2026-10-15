"""Train a SAC agent on the practised order (arm26, six muscles, start at the ring centre).

    python train_agent.py --seed 7 --envs 1 --out runs/agent1      (agent 1)
    python train_agent.py --seed 101 --envs 4 --out runs/agent2    (agent 2)
    python train_agent.py --seed 202 --envs 4 --out runs/agent3    (agent 3)

Every 10k transitions the policy runs 10 deterministic episodes; best_model.zip keeps the highest (success, -path
ratio). Training stops early after 3 evaluations in a row with success >= 0.95, path ratio <= 1.2 and deviation <= 2 cm.
"""
import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import arm

PATH_PENALTY = 0.05
EVALUATE_EVERY = 10_000
EPISODES = 10
SUCCESS, PATH_RATIO, DEVIATION, PATIENCE = 0.95, 1.2, 0.02, 3


def make_worker(out, rank):
    env = arm.make_env(start=arm.START, path_penalty=PATH_PENALTY)
    return Monitor(env, filename=str(out / f'worker_{rank}'), info_keywords=('is_success',))


def line_deviation(point, start, goal):
    direction = goal - start
    scale = float(direction @ direction)
    projection = 0.0 if scale == 0 else np.clip((point - start) @ direction / scale, 0, 1)
    return float(np.linalg.norm(point - (start + projection * direction)))


def evaluate(model, env):
    """Success, mean path ratio and largest deviation from the straight line of each completed reach."""
    episodes = []
    for episode in range(EPISODES):
        obs, info = env.reset(seed=episode)
        start, goal = np.asarray(info['states']['fingertip'], np.float32), np.asarray(info['goal'], np.float32)
        length, deviation, ratios, deviations = 0.0, 0.0, [], []
        for _ in range(env.task.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            point = np.asarray(info['states']['fingertip'], np.float32)
            length += float(info['step_distance'])
            deviation = max(deviation, line_deviation(point, start, goal))
            if bool(info['completed']):
                ratios.append(length / max(float(np.linalg.norm(goal - start)), 1e-8))
                deviations.append(deviation)
                start, goal = point, np.asarray(info['goal'], np.float32)
                length = deviation = 0.0
            if terminated or truncated:
                break
        episodes.append((bool(info['finished']), ratios, deviations))
    success = float(np.mean([e[0] for e in episodes]))
    ratios = [np.mean(e[1]) for e in episodes if e[1]]
    deviations = [max(e[2]) for e in episodes if e[2]]
    return success, float(np.mean(ratios)) if ratios else np.inf, float(max(deviations)) if deviations else np.inf


class Evaluation(BaseCallback):
    def __init__(self, every, out, progress, bar):
        super().__init__()
        self.every, self.out, self.progress, self.bar = every, out, progress, bar
        self.best, self.streak, self.history = (-1.0, -np.inf), 0, []

    def _init_callback(self):
        self.env = arm.make_env(start=arm.START, path_penalty=PATH_PENALTY)

    def _evaluate(self):
        success, ratio, deviation = evaluate(self.model, self.env)
        self.history.append(dict(step=self.num_timesteps, success=success, path_ratio=ratio, deviation=deviation))
        (self.out / 'evaluations.json').write_text(json.dumps(self.history, indent=1))
        if (success, -ratio) > self.best:
            self.best = (success, -ratio)
            self.model.save(str(self.out / 'best_model'))
        self.streak = self.streak + 1 if success >= SUCCESS and ratio <= PATH_RATIO and deviation <= DEVIATION else 0
        self.progress.update(self.bar, description=f'SAC (eval success {success:.2f}, best {self.best[0]:.2f})')

    def _on_step(self):
        self.progress.update(self.bar, completed=self.num_timesteps)
        if self.n_calls % self.every:
            return True
        self._evaluate()
        return self.streak < PATIENCE

    def _on_training_end(self):
        self._evaluate()


def train(seed, envs, steps, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    workers = [partial(make_worker, out, rank) for rank in range(envs)]
    env = DummyVecEnv(workers) if envs == 1 else SubprocVecEnv(workers, start_method='spawn')
    model = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=1_000_000, learning_starts=10_000, batch_size=256,
                gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1 if envs == 1 else -1, ent_coef='auto', seed=seed, device='cpu', verbose=0)
    columns = (TextColumn('{task.description}'), BarColumn(), TextColumn('{task.completed:,.0f}/{task.total:,.0f}'),
               TimeElapsedColumn(), TimeRemainingColumn())
    with Progress(*columns) as progress:
        bar = progress.add_task('SAC', total=steps)
        model.learn(total_timesteps=steps, callback=Evaluation(EVALUATE_EVERY // envs, out, progress, bar))
    model.save(str(out / 'last_model'))
    (out / 'config.json').write_text(json.dumps(dict(
        seed=seed, envs=envs, steps=steps, transitions=int(model.num_timesteps), sequence=list(arm.PRACTISED),
        start_deg=list(arm.START), reward=dict(progress=1.0, distance=0.1, path_penalty=PATH_PENALTY, target=20.0,
                                               completion=200.0),
        sac=dict(learning_rate=3e-4, buffer_size=1_000_000, learning_starts=10_000, batch_size=256, gamma=0.99,
                 tau=0.005, updates_per_transition=1, ent_coef='auto')), indent=1))
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--envs', type=int, default=1, choices=(1, 2, 4, 5, 8, 10))
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--threads', type=int, default=6, help='learner threads when --envs > 1')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1 if args.envs == 1 else args.threads)
    train(args.seed, args.envs, args.steps, args.out)
