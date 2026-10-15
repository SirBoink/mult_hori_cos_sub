"""Two-joint, six-muscle arm (MotorNet arm26, Euler, 10 ms) reaching eight targets on a 10 cm ring in sequence."""
import gymnasium as gym
import motornet as mn
import numpy as np
import torch

PRACTISED = (0, 3, 7, 5, 6, 1, 2, 4)   # the order every agent was trained on
RING = 0.1                             # m, target ring radius
HIT_RADIUS = 0.02                      # m
ARRIVAL_SPEED = 0.1                    # m/s, a target counts as acquired inside HIT_RADIUS below this speed
START = (45., 90.)                     # deg, shoulder and elbow at the ring centre
STEPS = 800                            # 8 s per sequence


def as_array(value):
    return value.detach().cpu().numpy().copy() if torch.is_tensor(value) else np.asarray(value).copy()


def make_arm(upper=1., fore=1.):
    """arm26; `upper` scales both segments' mass and inertia with the forearm, `fore` the forearm alone."""
    muscle = mn.muscle.RigidTendonHillMuscle()
    if upper == 1 and fore == 1:
        return mn.effector.RigidTendonArm26(muscle=muscle, timestep=.01)
    skeleton = mn.skeleton.TwoDofArm(m1=1.82 * upper, m2=1.43 * upper * fore, l1g=.135, l2g=.165,
                                     i1=.051 * upper, i2=.057 * upper * fore, l1=.309, l2=.333)
    return mn.effector.RigidTendonArm26(muscle=muscle, skeleton=skeleton, timestep=.01)


class SequenceTask(mn.environment.Environment):
    """Targets in a given order. Observation: goal, one-hot target, vision, proprioception, muscle activations (30)."""

    target_names = tuple(f'T{i}' for i in range(8))
    reference_posture = np.deg2rad([45.0, 90.0]).astype(np.float32)

    def __init__(self, effector, sequence, *, hit_radius=HIT_RADIUS, ring=RING, max_episode_steps=STEPS,
                 path_penalty=0., random_start=False, start=None, action_noise=0., obs_noise=0., differentiable=False):
        self.default_sequence = tuple(int(t) for t in sequence)
        self.hit_radius = float(hit_radius)
        self.ring = float(ring)
        self.tolerance = 0.01
        self.max_episode_steps = int(max_episode_steps)
        self.distance_scale = 0.1
        self.path_penalty = float(path_penalty)
        self.random_start = bool(random_start)
        self.start = None if start is None else np.deg2rad(np.asarray(start, dtype=np.float32))
        super().__init__(effector=effector, name='SequenceTask', max_ep_duration=max_episode_steps * effector.dt,
                         action_noise=action_noise, obs_noise=obs_noise, proprioception_delay=None, vision_delay=None,
                         differentiable=differentiable)
        joint_state = torch.tensor([[*self.reference_posture, 0.0, 0.0]], device=self.device)
        center = self.joint2cartesian(joint_state)[:, :2].squeeze(0)
        angles = torch.deg2rad(torch.arange(8, dtype=torch.float32, device=self.device) * 45)
        targets = center + self.ring * torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
        self.register_buffer('target_center', center)
        self.register_buffer('targets', targets)
        self.register_buffer('target_joint_angles', self._joint_angles(targets))

    def _get_obs_size(self):
        return 2 + len(self.target_names) + 2 + 3 * self.effector.n_muscles + self.effector.n_muscles * self.action_frame_stacking

    def _joint_angles(self, targets):
        xy = targets.detach().cpu().numpy()
        l1, l2 = self.skeleton.L1, self.skeleton.L2
        elbow = np.arccos(np.clip((np.sum(xy ** 2, axis=1) - l1 ** 2 - l2 ** 2) / (2 * l1 * l2), -1, 1))
        shoulder = np.arctan2(xy[:, 1], xy[:, 0]) - np.arctan2(l2 * np.sin(elbow), l1 + l2 * np.cos(elbow))
        return torch.tensor(np.column_stack((shoulder, elbow)).astype(np.float32), device=self.device)

    def get_obs(self, action=None, deterministic=False):
        self.update_obs_buffer(action=action)
        physical = torch.cat([self.goal, self.obs_buffer['vision'][0], self.obs_buffer['proprioception'][0]]
                             + self.obs_buffer['action'][:self.action_frame_stacking], dim=-1)
        activation = self.states['muscle'][:, 0, :]
        if not deterministic:
            action_features = self.effector.n_muscles * self.action_frame_stacking
            physical = self.apply_noise(physical, noise=self.obs_noise[:2] + self.obs_noise[10:24 + action_features])
            activation = self.apply_noise(activation, noise=self.obs_noise[24 + action_features:30 + action_features])
        one_hot = torch.nn.functional.one_hot(self._sequence[self.sequence_index], num_classes=8).to(torch.float32)
        obs = torch.cat((physical[:, :2], one_hot, physical[:, 2:], activation), dim=-1)
        return obs if self.differentiable else self.detach(obs)

    def reset(self, *, seed=None, options=None):
        self._set_generator(seed=seed)
        options = {} if options is None else dict(options)
        posture = self.reference_posture
        if self.start is not None:
            posture = self.start
        elif self.random_start:
            first = int(options.get('sequence', self.default_sequence)[0])
            start_target = int(self.np_random.choice([t for t in range(8) if t != first]))
            posture = self.detach(self.target_joint_angles[start_target])
        batch_size = int(options.get('batch_size', 1))
        self.sequence = tuple(int(t) for t in options.get('sequence', self.default_sequence))
        self.elapsed_steps = 0
        self._sequence = torch.tensor(self.sequence, dtype=torch.long, device=self.device)
        self.sequence_index = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.hold_count = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        deterministic = options.get('deterministic', False)
        joint_state = torch.tensor(posture, device=self.device).repeat(batch_size, 1)
        _, info = super().reset(seed=None, options=dict(batch_size=batch_size, joint_state=joint_state,
                                                        deterministic=deterministic))
        self.goal = self.targets[self._sequence[0]].repeat(batch_size, 1)
        obs = self.get_obs(deterministic=deterministic)
        info.update(self._task_info())
        return obs, info

    def step(self, action, deterministic=False, **kwargs):
        self.elapsed_steps += 1
        self.elapsed = self.elapsed_steps * self.dt
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        noisy_action = action if deterministic else self.apply_noise(action, self.action_noise)
        goal = self.goal.clone()
        active = ~self.finished
        before_hand = self.states['fingertip'].clone()
        before_distance = torch.linalg.vector_norm(before_hand - goal, dim=1)
        self.effector.step(noisy_action, **kwargs)

        after_hand = self.states['fingertip']
        after_distance = torch.linalg.vector_norm(after_hand - goal, dim=1)
        progress = before_distance - after_distance
        step_distance = torch.linalg.vector_norm(after_hand - before_hand, dim=1)
        wasted = torch.clamp(step_distance - progress, min=0.0)
        zero = lambda value: torch.where(active, value, torch.zeros_like(value))
        progress_reward = zero(progress / self.tolerance)
        path_penalty = zero(self.path_penalty * wasted / self.tolerance)
        distance_reward = zero(-self.distance_scale * after_distance / self.ring)
        speed = step_distance / self.dt

        completed = (after_distance <= self.hit_radius) & (speed <= ARRIVAL_SPEED) & active
        last = self.sequence_index == len(self.sequence) - 1
        self.finished |= completed & last
        self.sequence_index += (completed & ~last).long()
        self.goal = self.targets[self._sequence[self.sequence_index]]

        obs = self.get_obs(action=noisy_action, deterministic=deterministic)
        reward = (progress_reward + distance_reward - path_penalty + 20. * completed.float()
                  + 200. * (completed & last).float())
        terminated = bool(torch.all(self.finished).item())
        truncated = bool(self.elapsed_steps >= self.max_episode_steps and not terminated)
        info = {'states': self._maybe_detach_states(), 'completed': self.detach(completed),
                'step_distance': self.detach(step_distance), 'progress_reward': self.detach(progress_reward),
                'path_penalty': self.detach(path_penalty), **self._task_info()}
        return obs, self.detach(reward[:, None]), terminated, truncated, info

    def _task_info(self):
        values = {'goal': self.goal, 'sequence_index': self.sequence_index, 'target': self._sequence[self.sequence_index],
                  'hold_count': self.hold_count, 'finished': self.finished,
                  'activation': self.states['muscle'][:, 0, :]}
        return values if self.differentiable else {key: self.detach(value) for key, value in values.items()}


class SequenceGym(gym.Env):
    """One task row as a Gymnasium environment; policy actions in [-1, 1] become excitations in [0, 1]."""

    metadata = {'render_modes': []}

    def __init__(self, task):
        self.task = task
        self.observation_space = task.observation_space
        self.action_space = gym.spaces.Box(-1.0, 1.0, task.action_space.shape, np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {}, batch_size=1)
        observation, info = self.task.reset(seed=seed, options=options)
        info = squeeze(info)
        info['is_success'] = False
        return np.asarray(observation[0], dtype=np.float32), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        excitation = (np.clip(action, -1.0, 1.0) + 1.0) / 2.0
        observation, reward, terminated, truncated, info = self.task.step(excitation[None, :])
        info = squeeze(info)
        info['is_success'] = bool(info['finished'])
        return np.asarray(observation[0], dtype=np.float32), float(reward[0, 0]), bool(terminated), bool(truncated), info


def squeeze(value):
    if isinstance(value, dict):
        return {key: squeeze(item) for key, item in value.items()}
    if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == 1:
        return value[0]
    return value


def make_env(sequence=PRACTISED, random_start=False, start=None, path_penalty=0., max_episode_steps=STEPS):
    """Single-row environment used for training."""
    task = SequenceTask(make_arm(), sequence, random_start=random_start, start=start, path_penalty=path_penalty,
                        max_episode_steps=max_episode_steps)
    return SequenceGym(task)


def make_task(sequence, postures, max_steps=STEPS, ring=RING, upper=1., fore=1.):
    """One task with a row per initial posture (deg), noise off; returns (task, observations)."""
    task = SequenceTask(make_arm(upper, fore), sequence, ring=ring, max_episode_steps=max_steps, start=START)
    postures = np.asarray(postures, dtype=np.float32)
    task.reset(seed=270000, options={'batch_size': len(postures), 'deterministic': True})
    joint = torch.tensor(np.column_stack((np.deg2rad(postures), np.zeros_like(postures))), dtype=torch.float32)
    task.effector.reset(options={'batch_size': len(postures), 'joint_state': joint})
    zero = torch.zeros((len(postures), 6))
    task.obs_buffer['vision'] = [task.get_vision().clone() for _ in task.obs_buffer['vision']]
    task.obs_buffer['proprioception'] = [task.get_proprioception().clone() for _ in task.obs_buffer['proprioception']]
    task.obs_buffer['action'] = [zero.clone() for _ in task.obs_buffer['action']]
    observation = np.asarray(task.get_obs(action=zero, deterministic=True), dtype=np.float32)
    return task, observation


def features(task):
    """phi = [x, y, vx, vy, six muscle activations]."""
    return np.column_stack((as_array(task.states['cartesian']), as_array(task.states['muscle'][:, 0, :])))


def targets(ring=RING):
    task, _ = make_task(PRACTISED, [START], ring=ring)
    return as_array(task.targets)
