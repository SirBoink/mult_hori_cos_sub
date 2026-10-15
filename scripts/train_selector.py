"""Semi-Markov double DQN that chooses, at every free step, the frozen agent or an option to one of the subgoals.

    python train_selector.py --agent 1 --subgoals data/subgoals/agent1.json --seed 1 --steps 1000000 --out runs/agent1_seed1
    python train_selector.py --agent 3 --subgoals data/subgoals/agent3_multiscale.json --seed 1 --steps 500000 \
        --random-orders --out runs/agent3_multiscale_seed1

Without --random-orders the selector trains on the practised order only; the kept checkpoint is the best on one
deterministic practised episode. With --random-orders it trains on random orders from random start targets, never on the
held-out or validation orders; the kept checkpoint is the best on the 16 validation orders x 10 starts.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

import arm
import control

GAMMA = 0.99
REPLAY = 100_000
WARMUP = 1000
BATCH = 64
TARGET_SYNC = 1000          # updates between target-network copies
EVALUATE_EVERY = 50_000     # primitive steps
LEARNING_RATE = 3e-4
GRADIENT_CLIP = 10.0
EPSILON_START, EPSILON_END, EPSILON_FRACTION = 1.0, 0.05, 0.5
DELIBERATION_COST = 0.1     # subtracted from the return of every decision
RESET_SEED = 7              # the practised episode used for checkpoint selection
ROOT = Path(__file__).resolve().parent


def epsilon(step, total):
    return EPSILON_END + (EPSILON_START - EPSILON_END) * max(1. - float(step) / (EPSILON_FRACTION * total), 0.)


def choose(values, mask, exploration, rng):
    """Greedy over available choices; with probability `exploration`, uniform over them."""
    if rng.random() >= exploration:
        return int(np.argmax(np.where(mask, values, -np.inf)))
    return int(rng.choice(np.flatnonzero(mask)))


def random_order(rng, excluded):
    while True:
        order = (arm.PRACTISED[0], *map(int, rng.permutation(arm.PRACTISED[1:])))
        if order not in excluded:
            return order


def run_episode(env, agent, reacher, subgoals, selector, rng, *, step=0, total=1, training=False,
                reset_seed=RESET_SEED, sequence=arm.PRACTISED, budget=None):
    observation, info = env.reset(seed=reset_seed, options={'sequence': sequence, 'deterministic': True})
    transitions, choices, total_reward = [], [], 0.
    while True:
        mask = control.available_options(info, subgoals)
        with torch.no_grad():
            values = selector(torch.as_tensor(observation[None], dtype=torch.float32)).numpy()[0]
        choice = choose(values, mask, epsilon(step, total) if training else 0., rng)
        result = control.run_choice(choice, observation, info, env, agent, reacher, subgoals,
                                    None if budget is None else int(budget - step))
        next_observation = np.asarray(result['observation'], dtype=np.float32)
        done = bool(result['terminated'] or result['truncated'])
        next_mask = np.ones(len(subgoals) + 1, dtype=bool) if done else control.available_options(result['info'], subgoals)
        reward = sum(r * GAMMA ** k for k, r in enumerate(result['rewards'])) - DELIBERATION_COST
        transitions.append((np.asarray(observation, dtype=np.float32), choice, reward, result['duration'],
                            next_observation, next_mask, done))
        choices.append(choice)
        total_reward += sum(result['rewards'])
        step += result['duration']
        observation, info = next_observation, result['info']
        if done or (budget is not None and step >= budget):
            break
    finished = bool(np.asarray(info['finished']).reshape(-1)[0])
    index = int(np.asarray(info['sequence_index']).reshape(-1)[0])
    episode = dict(success=finished, targets_completed=index + int(finished), steps=int(env.task.elapsed_steps),
                   return_=float(total_reward), options=int(sum(c > 0 for c in choices)))
    return transitions, episode, step


def update(online, target, optimizer, replay):
    states, actions, rewards, durations, next_states, masks, done = zip(*random.sample(replay, BATCH))
    states = torch.as_tensor(np.stack(states), dtype=torch.float32)
    next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.long)
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    durations = torch.as_tensor(durations, dtype=torch.float32)
    masks = torch.as_tensor(np.stack(masks), dtype=torch.bool)
    done = torch.as_tensor(done, dtype=torch.bool)
    values = online(states).gather(1, actions[:, None]).squeeze(1)
    with torch.no_grad():
        choices = online(next_states).masked_fill(~masks, -torch.inf).argmax(dim=1)
        next_values = target(next_states).gather(1, choices[:, None]).squeeze(1)
        targets = rewards + torch.pow(torch.as_tensor(GAMMA), durations) * next_values * (~done)
    loss = torch.nn.functional.smooth_l1_loss(values, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online.parameters(), GRADIENT_CLIP)
    optimizer.step()


def validation_scorer(agent, reacher, subgoals, panel):
    """Success and targets completed of the current selector on validation orders x 10 starts."""
    def score(selector):
        rows = []
        with torch.no_grad():
            for o, order in enumerate(panel['orders']):
                starts = control.start_postures(panel['postures_deg'], o)
                task, obs = arm.make_task(order, [s['q'] for s in starts], panel['max_steps'])
                rows += control.transfer(task, obs, agent, reacher, 'selector', subgoals, selector, panel['max_steps'])[0]
        return np.mean([r['success'] for r in rows]), np.mean([r['targets_completed'] for r in rows])
    return score


def train(out, agent_path, subgoals_path, steps, seed, random_orders=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    subgoals = control.load_subgoals(subgoals_path)
    agent = control.load_policy(agent_path)
    reacher = control.load_policy(ROOT / 'checkpoints/reacher.zip', 22)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = arm.make_env(random_start=random_orders)
    eval_env = arm.make_env() if random_orders else env
    score_validation, excluded = None, ()
    if random_orders:
        panels = ROOT / 'data/panels'
        validation = json.loads((panels / 'orders16_validation.json').read_text())
        heldout = json.loads((panels / 'orders96_heldout.json').read_text())
        score_validation = validation_scorer(agent, reacher, subgoals, validation)
        excluded = tuple(tuple(o) for o in heldout['orders']) + tuple(tuple(o) for o in validation['orders'])
    count = len(subgoals) + 1
    online, target = control.QNetwork(30, count), control.QNetwork(30, count)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=LEARNING_RATE)
    config = dict(agent=Path(agent_path).resolve().relative_to(ROOT).as_posix(), reacher='checkpoints/reacher.zip', subgoals=Path(subgoals_path).as_posix(), state_dim=30,
                  action_count=count, hidden_units=[128, 128], steps=int(steps), seed=int(seed),
                  orders='random' if random_orders else 'practised',
                  checkpoint='best on 16 validation orders' if random_orders else 'best on the practised order',
                  discount=GAMMA, deliberation_cost=DELIBERATION_COST, exploration='uniform over available choices',
                  epsilon=dict(start=EPSILON_START, end=EPSILON_END, fraction=EPSILON_FRACTION),
                  option_timeout_steps=control.OPTION_STEPS, termination_radius_m=control.SUBGOAL_RADIUS)
    (out / 'config.json').write_text(json.dumps(config, indent=1))
    frozen = control.QNetwork(30, count)
    for parameter in frozen.parameters():
        parameter.data.zero_()
    _, reference, _ = run_episode(eval_env, agent, reacher, subgoals, frozen, rng, total=steps)   # frozen agent alone
    log = csv.writer((out / 'evaluations.csv').open('w', newline=''))
    log.writerow(['step', 'success', 'targets_completed', 'steps', 'options', 'validation_success', 'validation_targets'])
    replay, position, updates, primitive, best, next_eval, episodes = [], 0, 0, 0, None, EVALUATE_EVERY, 0
    columns = (TextColumn('{task.description}'), BarColumn(), TextColumn('{task.completed:,.0f}/{task.total:,.0f} steps'),
               TimeElapsedColumn(), TimeRemainingColumn())
    with Progress(*columns) as progress:
        bar = progress.add_task(f'selector seed {seed} (frozen agent: {reference["steps"]} steps)', total=steps)
        while primitive < steps:
            sequence = random_order(rng, excluded) if random_orders else arm.PRACTISED
            transitions, _, primitive = run_episode(
                env, agent, reacher, subgoals, online, rng, step=primitive, total=steps, training=True,
                reset_seed=seed + episodes + 1 if random_orders else RESET_SEED, sequence=sequence, budget=steps)
            episodes += 1
            for transition in transitions:
                if len(replay) < REPLAY:
                    replay.append(transition)
                else:
                    replay[position] = transition
                    position = (position + 1) % REPLAY
                if len(replay) >= WARMUP and len(replay) >= BATCH:
                    update(online, target, optimizer, replay)
                    updates += 1
                    if updates % TARGET_SYNC == 0:
                        target.load_state_dict(online.state_dict())
            progress.update(bar, completed=min(primitive, steps))
            if primitive >= next_eval or primitive >= steps:
                _, evaluation, _ = run_episode(eval_env, agent, reacher, subgoals, online, rng, total=steps)
                validation = score_validation(online) if random_orders else (np.nan, np.nan)
                score = validation if random_orders else (int(evaluation['success']), evaluation['targets_completed'],
                                                          -evaluation['steps'], evaluation['return_'])
                log.writerow([primitive, evaluation['success'], evaluation['targets_completed'], evaluation['steps'],
                              evaluation['options'], *validation])
                if best is None or score > best:
                    best = score
                    torch.save({'model': online.state_dict(), 'config': config}, out / 'best.pt')
                while next_eval <= primitive:
                    next_eval += EVALUATE_EVERY
    torch.save({'model': online.state_dict(), 'config': config}, out / 'last.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--agent', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--subgoals', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--random-orders', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    train(args.out, ROOT / f'checkpoints/agent{args.agent}.zip', args.subgoals,
          args.steps, args.seed, args.random_orders)
