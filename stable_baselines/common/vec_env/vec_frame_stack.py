import warnings

import numpy as np
from gym import spaces

from stable_baselines.common.vec_env.base_vec_env import VecEnvWrapper


class VecFrameStack(VecEnvWrapper):
    """
    Frame stacking wrapper for vectorized environment

    :param venv: (VecEnv) the vectorized environment to wrap
    :param n_stack: (int) Number of frames to stack
    """
    
    def __init__(self, venv, n_stack, n_offset):
        self.venv = venv
        self.n_stack = n_stack
        self.n_offset = n_offset
        wrapped_obs_space = venv.observation_space
        low = np.repeat(wrapped_obs_space.low, self.n_stack, axis=-1)
        high = np.repeat(wrapped_obs_space.high, self.n_stack, axis=-1)
        #self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        self.stackedobs = np.zeros((venv.num_envs, low.shape[0], low.shape[1], n_stack*n_offset), low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        last_ax_size = observations.shape[-1]
        self.stackedobs = np.roll(self.stackedobs, shift=-last_ax_size, axis=-1)
        for i, done in enumerate(dones):
            if done:
                if 'terminal_observation' in infos[i]:
                    old_terminal = infos[i]['terminal_observation']
                    new_terminal = np.concatenate(
                        (self.stackedobs[i, ..., :-last_ax_size], old_terminal), axis=-1)
                    infos[i]['terminal_observation'] = new_terminal
                else:
                    warnings.warn(
                        "VecFrameStack wrapping a VecEnv without terminal_observation info")
                self.stackedobs[i] = 0
        self.stackedobs[..., -observations.shape[-1]:] = observations
        test = self.stackedobs[:, :, :, self.n_offset-1::self.n_offset]
        return self.stackedobs[:, :, :, self.n_offset-1::self.n_offset], rewards, dones, infos
        # return self.stackedobs[:, :, :, -1::-self.n_offset], rewards, dones, infos

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        obs = obs[0,:,:,:]
        # obs = np.swapaxes(obs, 2, 1)
        # obs = np.swapaxes(obs, 1, 0)
        # for i in range(self.stackedobs.shape[3]):
        #     self.stackedobs[:,:,:,i] = obs
        return self.stackedobs[:, :, :, self.n_offset-1::self.n_offset]

    def close(self):
        self.venv.close()
