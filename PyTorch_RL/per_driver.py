import numpy as np
from typing import Any, List, Tuple, Optional, Callable, Union, NamedTuple
from copy import deepcopy
import torch as th
from sumTreeBatched import PrioritizedExperienceReplayBuffer as PER, stochastic_priority_replacement
import threading
import queue
import gymnasium as gym
from collections import deque
from enum import Enum

class ImplementationError(BaseException):
    pass

class Transition(NamedTuple):
    curr_obs: Any
    next_obs: Any
    action: Any
    reward: Any
    done: Any
    infos: Any

class PERSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    discounts: th.Tensor
    weights: th.Tensor
    indexes:th.Tensor

class PERCommands(Enum):
    add = 0
    update_priorities = 1
    update_beta = 2

class PERBetaScheduler:
    def __init__(self, beta_start, beta_end = 1.0):
        self.beta_start = np.float32(beta_start)
        self.beta_end = np.float32(beta_end)
    
    def __call__(self, progress_remaining):
        return self.beta_end * (self.beta_start/self.beta_end)**progress_remaining


class PERWithDriver:
    """
    A Prioritized Experience Replay (PER) buffer compatible with SB3's replay buffer API.
    It wraps an existing numpy-based PER implementation and adds:
        - N-step transition handling
        - Batched insertion
        - Optional TD-error computation
    """

    def __init__(
        self,
        buffer_size: int,                       # same as 'capacity' in your old PER
        observation_space,
        action_space,
        device: th.device,
        dtype: th.dtype = None,
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        n_step: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        per_policy: Optional[
            Callable[[Any, Tuple[Any, float]], np.ndarray]
        ] = stochastic_priority_replacement,
        gamma:float = 0.99,
        typical_episode_len: int = 390
    ):
        self.device = device
        self.dtype = dtype
        self.n_envs = n_envs
        self.optimize_memory_usage = optimize_memory_usage
        self.observation_space = observation_space
        self.action_space = action_space
        self.buffer_size = buffer_size
        self.n_step = n_step
        self.batch_size = 2 * typical_episode_len * n_envs

        self.gamma = gamma
        self.gammas = self.gamma ** np.arange(0,self.n_step,1)

        # Rolling window to accumulate N-step transitions
        self.n_step_transition = [
            deque(maxlen=n_step)
            for _ in range(n_envs)
        ] # N_env by N_step

        self.batched_transition = [None for _ in range(self.batch_size)]
        self.batch_len = 0 # number of Non None n step transitions in the batch


        # Underlying PER buffer that does the actual storage and sampling
        self.per_buffer = PER(
            capacity=buffer_size,
            alpha=alpha,
            beta=beta,
            replacement_policy=per_policy,
        )

        self.beta_schedule = PERBetaScheduler(beta, 1)

        # post initialization, these are initialized once the algorithm is initialized
        self.compute_td_error = None
        self.q_net = None
        self.q_net_target = None

        # Async PER insertion
        self._per_queue = queue.Queue(maxsize=100)
        self._per_thread = threading.Thread(target=self._per_worker, daemon=True)
        self._per_thread.start()

    def _per_worker(self):
        while True:
            command, payload = self._per_queue.get()
            try:
                if command == PERCommands.add:
                    self.per_buffer.add_batch_experience(payload)
                elif command == PERCommands.update_priorities:
                    indices, td_vals = payload
                    self.per_buffer.update_leaf_priorities(np.array(indices), np.array(td_vals))
                else:
                    raise ValueError(f"Unknown PER command: {command}")
            except Exception as e:
                print("PER worker error:", e)
            finally:
                self._per_queue.task_done()

    def _enqueue_per(self, command: PERCommands, payload: Any):
        """Add a task to the PER queue."""
        if self._per_queue.full():
            print("Queue is full")
            
        self._per_queue.put((command, payload))
   
    def add(self,
        obs: Union[np.ndarray, dict],      # shape: [n_env, obs_shape]
        next_obs: Union[np.ndarray, dict], # shape: [n_env, obs_shape]
        action: np.ndarray,                # shape: [n_env, act_shape]
        reward: np.ndarray,                # shape: [n_env,]
        done: np.ndarray,                  # shape: [n_env,]
        infos: List[dict[str, Any]],       # shape: [n_env,]
    ) -> None:
        """
        Insert one step of transitions (one per environment) into the N-step PER buffer.

        This function handles:
            - Constructing single-step transitions for each environment.
            - Maintaining per-environment N-step rolling windows.
            - Creating N-step transitions once enough steps are accumulated.
            - Flushing terminal transitions to the PER buffer with TD-error computation.
            - Accumulating a batch of transitions for efficient insertion into the underlying PER.

        Parameters
        ----------
        obs : np.ndarray or dict
            Current observations for all environments. Shape: [n_env, obs_shape].
        next_obs : np.ndarray or dict
            Next observations after taking actions. Shape: [n_env, obs_shape].
        action : np.ndarray
            Actions taken in each environment. Shape: [n_env, act_shape].
        reward : np.ndarray
            Rewards received after taking actions. Shape: [n_env,].
        done : np.ndarray
            Terminal flags indicating episode termination in each environment. Shape: [n_env,].
        infos : List[dict[str, Any]]
            Additional information per environment returned by the environment step. Length: n_env.

        Assumptions
        -----------
        1. `n_step` is strictly smaller than the typical episode length, so that
        when a terminal transition is encountered, there are already `n_step` transitions
        available for creating an N-step target.
        2. The batch buffer (`self.batched_transition`) is preallocated large enough to
        accommodate at least one full episode for each environment. This ensures
        that `self.batch_len` will not exceed its capacity, preventing index errors.
        3. The `loss_fn` attribute must be assigned before calling `add()`. It is used to
        compute TD errors for the batch before inserting into the PER buffer.

        Notes
        -----
        - Per-environment N-step transitions are maintained independently.
        - The method flushes transitions to the PER buffer whenever an environment reaches
        a terminal state, computing TD errors for the accumulated batch first.
        - The batch buffer continues accumulating transitions until flush, ensuring
        efficient insertion into the underlying PER.

        """
        
        for env_idx in range(self.n_envs):
            # === Local aliases for readability ===
            n_step_queue = self.n_step_transition[env_idx]

            # === Step 0: get curr and next_obs for both formats
            if isinstance(obs, dict):
                curr_obs = {k: obs[k][env_idx] for k in obs.keys()}
                next_obs = {k: next_obs[k][env_idx] for k in next_obs.keys()}
            else:
                curr_obs = obs[env_idx]
                next_obs = next_obs[env_idx]


            # === Step 1: Build single-step transition for this environment ===
            transition = Transition(
                curr_obs = curr_obs,
                action   = action[env_idx],
                reward   = reward[env_idx],
                next_obs = next_obs,
                done     = done[env_idx],
                infos    = infos[env_idx],
            )
            # === Step 2: Maintain rolling N-step window ===
            # If the N-step buffer for this environment is full, remove the oldest transition.


            # Append the new transition and update the count.
            n_step_queue.append(transition)


            # === Step 3: Skip until we have at least N transitions ===
            if len(n_step_queue) < self.n_step:continue

            # === Step 4: Build the N-step transition ===
            n_step_data = self._make_n_step_transition(list(n_step_queue))

            # === Step 5: Handle episode termination ===
            if transition.done:
                # Flush remaining N-step transitions for this environment.
                for _ in range(self.n_step):
                    assert n_step_data is not None
                    self.batched_transition[self.batch_len] = n_step_data
                    self.batch_len += 1

                    # Slide window forward by one step and mark empty slots.
                    n_step_queue.append(None)
                    n_step_data = self._make_n_step_transition(list(n_step_queue))

                # Verify the entire N-step buffer is cleared.
                assert all([ t == None for t in  list(n_step_queue)]), \
                    "Internal error: N-step window not fully cleared after episode end."

                n_step_queue.clear() # reset the queue

                # === Step 6: Compute TD errors and add to PER ===
                if self.compute_td_error is None:
                    raise ImplementationError("Loss function not assigned!")

                # Compute losses and TD errors for accumulated transitions.
                transition = self.transitions_to_tensors(self.batched_transition[:self.batch_len])
                samples = PERSamples(
                    observations = transition.curr_obs,
                    actions = transition.action,
                    next_observations = transition.next_obs,
                    dones = transition.done,
                    rewards = transition.reward,
                    discounts=th.tensor(self.gamma ** self.n_step, dtype=self.dtype, device=self.device),
                    weights=None,
                    indexes=None
                ) # weights and indexes are not needed for td_target computation.
                
                td_errors = self.compute_td_error(samples, self.q_net, self.q_net_target, training = False).cpu().numpy()

                # Create list of (transition, td_error) tuples.
                input_tuple_list = [
                    (self.batched_transition[i], td_errors[i])
                    for i in range(self.batch_len)
                ]
                
                # Add to underlying PER buffer, asynchronously. 
                self._enqueue_per(PERCommands.add, input_tuple_list)

                # Reset the batch buffer after flush.
                self.batch_len = 0

            else:
                # === Step 7: Continue accumulating transitions until flush ===
                self.batched_transition[self.batch_len] = n_step_data
                self.batch_len += 1
            
    def _make_n_step_transition(self, transitions: List[Transition]):
        """
        Build a single N-step transition from a list of consecutive single-step transitions.

        Parameters
        ----------
        transitions : List[Transition]
            List of consecutive transitions (length N), may contain None for empty slots.

        Returns
        -------
        Transition
            A single N-step transition with:
            - `curr_obs` and `action` from the first step
            - `next_obs` and `infos` from the last valid step
            - `reward` equal to the discounted sum of rewards over N steps, masked by any terminal (`done`) flags
            - `done` set to True if any step in the sequence was terminal, False otherwise

        Notes
        -----
        - Handles partial windows by ignoring None entries.
        - Uses `self.gammas` for discounting.
        - Ensures that bootstrapping does not occur past episode termination.
        """
        if len(transitions) != self.n_step:
            raise ImplementationError(f"transitions should be {self.n_step}-step but are instead {len(transitions)}-step")
        
        # get the Non None transition
        transitions = [t for t in transitions if t is not None]

        num_non_nones = len(transitions)

        if num_non_nones == 0:
            return None # base case

        obs, action = transitions[0].curr_obs, transitions[0].action
        rews = np.array([t.reward for t in transitions])
        not_dones = np.array([ 1 - np.float32(t.done) for t in transitions]) # convert bool to float
        not_dones = np.cumprod(not_dones)
        total_reward = np.sum(rews * not_dones * self.gammas[:num_non_nones]) # indexing is exclusive 0: non-none - 1

        next_obs = transitions[num_non_nones - 1].next_obs
        done = bool( 1 - not_dones[num_non_nones - 1] )
        infos = transitions[num_non_nones - 1].infos

        return Transition(
            curr_obs = obs,
            action = action,
            next_obs=next_obs,
            reward=total_reward,
            done=done,
            infos=infos
            )

    def post_initialization(self,
            compute_td_error:Callable[[List[Transition]], np.ndarray],
            q_net:th.nn.Module,
            q_net_target:th.nn.Module):
        
        self.compute_td_error =  compute_td_error
        self.q_net =  q_net
        self.q_net_target = q_net_target

    def sample(self, batch_size: int) -> PERSamples:
        """
        Sample a batch from PER buffer and return a PERSamples namedtuple.
        All tensors are moved to `self.device` with dtype `self.dtype`, except `indexes` which stays on CPU.
        """
        # print(f"Number of actual samples in PER: {self.per_buffer.length}")
        # print(f"Per Capacity: {self.per_buffer.capacity}")
        # print(f"PER filled (percent): {self.per_buffer.length / self.per_buffer.capacity * 100}")
        # Sample from underlying PER buffer
        transitions: List[Transition]
        transitions, weights, indexes = self.per_buffer.sample(batch_size)

        transition = self.transitions_to_tensors(transitions)

        # Importance sampling weights
        weights = th.tensor(weights, device=self.device, dtype=self.dtype)

        # indexes stay on CPU
        indexes = th.tensor(indexes, device='cpu', dtype=th.int32)

        return PERSamples(
            observations = transition.curr_obs,
            actions = transition.action,
            next_observations = transition.next_obs,
            dones = transition.done,
            rewards = transition.reward,
            discounts=th.tensor(self.gamma ** self.n_step, dtype=self.dtype, device=self.device),
            weights=weights,
            indexes=indexes
        )

    def update_priorities(self, indices, td_errors):
        payload = (indices, td_errors.detach().cpu().numpy().squeeze(-1))
        self._enqueue_per(PERCommands.update_priorities, payload)

    def update_beta(self,progress_remaining:float):
        self.per_buffer.set_beta(
            self.beta_schedule(progress_remaining)
        )

    def transitions_to_tensors(self,transitions):
        """
        Converts a list of transitions into PyTorch tensors.

        Args:
            transitions: list of transition objects with attributes
                        curr_obs, next_obs, action, reward, done

        Returns:
            Transition( obs, next_obs, actions, rewards, dones), without info
            obs and next_obs are either tensors or dicts of tensors
        """
        # Observations and next observations
        if isinstance(self.observation_space, gym.spaces.Dict):
            obs = {
                key: th.tensor(
                    np.stack([t.curr_obs[key] for t in transitions], axis=0),
                    device=self.device,
                    dtype=self.dtype
                )
                for key in self.observation_space.spaces
            }
            next_obs = {
                key: th.tensor(
                    np.stack([t.next_obs[key] for t in transitions], axis=0),
                    device=self.device,
                    dtype=self.dtype
                )
                for key in self.observation_space.spaces
            }
        else:
            obs = th.tensor(
                np.stack([t.curr_obs for t in transitions], axis=0),
                device=self.device,
                dtype=self.dtype
            )
            next_obs = th.tensor(
                np.stack([t.next_obs for t in transitions], axis=0),
                device=self.device,
                dtype=self.dtype
            )

        # Actions, rewards, dones
        actions = th.tensor(
            np.stack([t.action for t in transitions], axis=0),
            device=self.device,
            dtype=self.dtype
        )
        rewards = th.tensor(
            np.stack([t.reward for t in transitions], axis=0),
            device=self.device,
            dtype=self.dtype
        )
        dones = th.tensor(
            np.stack([t.done for t in transitions], axis=0),
            device=self.device,
            dtype=self.dtype
        )

        return Transition(
            curr_obs=obs,
            next_obs= next_obs,
            action=actions,
            reward=rewards,
            done=dones,
            infos=None
            )

