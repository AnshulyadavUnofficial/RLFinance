from tf_agents.utils import eager_utils,  nest_utils, common
from tf_agents.agents import tf_agent
from tf_agents.agents.dqn.dqn_agent import DqnLossInfo
from typing import cast
from tf_agents.specs import tensor_spec
from tf_agents.typing import types
from tf_agents.trajectories import TimeStep, Trajectory, StepType
from tf_agents.networks import Network, utils as network_utils
from tf_agents.policies import q_policy, boltzmann_policy, greedy_policy, epsilon_greedy_policy
import tensorflow as tf


from typing import Optional, Text, Dict

class R2D2Agent(tf_agent.TFAgent):
    
    def __init__(
        self,
        time_step_spec: TimeStep,
        action_spec: types.NestedTensorSpec,
        q_network: Network,
        optimizer: types.Optimizer,
        observation_and_action_constraint_splitter: Optional[types.Splitter] = None,
        epsilon_greedy: Optional[types.FloatOrReturningFloat] = 0.1, # type: ignore
        boltzmann_temperature: Optional[types.FloatOrReturningFloat] = None, # type: ignore
        # R2D2-Specific Parameters
        n_step_update: int = 5,
        burn_in: int = 0,
        retrace_lambda: float = 0.8,
        # Params for target Updates
        target_q_network: Optional[Network] = None,
        target_update_tau: types.Float = 1.0, # type: ignore
        target_update_period: int = 100,
        # Params for training
        td_errors_loss_fn: Optional[types.LossFn] = None,
        gamma: types.Float = 0.99, # type: ignore
        reward_scale_factor: types.Float = 1.0, # type: ignore
        gradient_clipping: Optional[types.Float] = None, # type: ignore
        # Params for debugging
        debug_summaries: bool = False,
        summarize_grads_and_vars: bool = False,
        train_step_counter: Optional[tf.Variable] = None,
        name: Optional[str] = "R2D2Agent",
    ):
        """Creates an R2D2 Agent.

        Args:
            time_step_spec: A `TimeStep` spec of the expected time_steps.
            action_spec: A nest of BoundedTensorSpec representing the actions.
            q_network: A `tf_agents.network.Network` to be used by the agent. Must be
                a stateful network (RNN). The network will be called with 
                `call(observation, step_type, network_state)` and should emit logits 
                over the action space along with updated network state.
            optimizer: The optimizer to use for training.
            observation_and_action_constraint_splitter: A function used to process
                observations with action constraints. 
            epsilon_greedy: Probability of choosing a random action in the default
                epsilon-greedy collect policy. Mutually exclusive with
                `boltzmann_temperature`.
            boltzmann_temperature: Temperature value for Boltzmann sampling. Mutually
                exclusive with `epsilon_greedy`.
            n_step_update: The number of steps to consider when computing TD error.
                Requires trajectories with time dimension >= burn_in + n_step + 1.
            burn_in: Number of initial steps in each trajectory used only for RNN 
                state initialization, not for loss calculation.
            retrace_lambda: Interpolation parameter for Retrace off-policy correction.
                Controls how much multi-step returns are weighted by truncated importance sampling coefficients:
                - A value of 0 reduces to standard one-step updates (no trace).
                - A value of 1 applies full multi-step Retrace corrections.
                Intermediate values balance bias and variance by scaling eligibility traces.
                Choosing a higher lambda can improve sample efficiency but may introduce more variance.
            target_q_network: (Optional.) Network to use as target network during
                Q learning. If None, a copy of `q_network` will be created.
            target_update_tau: Factor for soft update of target networks.
            target_update_period: Period for soft update of target networks.
            td_errors_loss_fn: Function for computing TD errors loss. Defaults to
                Huber loss.
            gamma: Discount factor for future rewards.
            reward_scale_factor: Multiplicative scale for rewards.
            gradient_clipping: Norm length to clip gradients.
            debug_summaries: Whether to gather debug summaries.
            summarize_grads_and_vars: Whether to log gradient and variable summaries.
            train_step_counter: Optional counter to increment every training step.
            name: The name of this agent.

        Raises:
            ValueError: For invalid combinations of parameters
            ValueError: If q_network is not stateful (not an RNN)
            ValueError: If action spec has invalid shape
            ValueError: If network outputs don't match action spec
        """
        tf.Module.__init__(self, name=name)
        
        # Enforce RNN requirement
        # if not q_network.state_spec:
        #     raise ValueError("R2D2 requires a stateful q_network (RNN).")

        action_spec = tensor_spec.from_spec(action_spec)
        self._check_action_spec(action_spec)
        
        if epsilon_greedy is not None and boltzmann_temperature is not None:
            raise ValueError(
                'Configured both epsilon_greedy value {} and temperature {}, '
                'however only one of them can be used for exploration.'.format(
                    epsilon_greedy, boltzmann_temperature))

        self._observation_and_action_constraint_splitter = (observation_and_action_constraint_splitter)
        self._q_network = q_network
        net_observation_spec = time_step_spec.observation
        if observation_and_action_constraint_splitter:
            net_observation_spec, _ = observation_and_action_constraint_splitter(net_observation_spec)
            
        q_network.create_variables(net_observation_spec)
        if target_q_network:
            target_q_network.create_variables(net_observation_spec)
            
        self._target_q_network = common.maybe_copy_target_network_with_checks(
            self._q_network, target_q_network, input_spec=net_observation_spec,
            name='TargetQNetwork')

        self._epsilon_greedy = epsilon_greedy
        self._n_step_update = n_step_update
        self._burn_in = burn_in
        self._retrace_lambda = retrace_lambda
        self._train_sequence_length = self._burn_in + self._n_step_update + 1
        self._boltzmann_temperature = boltzmann_temperature
        self._optimizer = optimizer
        self._td_errors_loss_fn = (td_errors_loss_fn or common.element_wise_huber_loss)
        self._gamma = gamma
        self._reward_scale_factor = reward_scale_factor
        self._gradient_clipping = gradient_clipping
        
        self._update_target = self._get_target_updater(
            target_update_tau, target_update_period)
        
        
        # Make the γ powers: [1, γ, γ², …, γⁿ⁻¹]
        self._gamma_powers =  tf.math.pow(gamma, tf.range(self._n_step_update, dtype=tf.float32))  # [N]

        # Setup Policies
        policy = q_policy.QPolicy(time_step_spec, action_spec,
            q_network=self._q_network,
            emit_log_probability=True, # needed for retrace lambda
            observation_and_action_constraint_splitter=(
                self._observation_and_action_constraint_splitter))

        if boltzmann_temperature is not None:
            collect_policy = boltzmann_policy.BoltzmannPolicy(policy, temperature=self._boltzmann_temperature)
        else:
            collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(policy, epsilon=self._epsilon_greedy)
            
        policy = greedy_policy.GreedyPolicy(policy)
        

        super().__init__(
            time_step_spec,
            action_spec,
            policy,
            collect_policy,
            train_sequence_length=self._train_sequence_length,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter,
            training_data_spec=None
        )
        

    def _check_action_spec(self, action_spec):
        flat_action_spec = tf.nest.flatten(action_spec)
        # TODO(oars): Get DQN working with more than one dim in the actions.
        if len(flat_action_spec) > 1 or flat_action_spec[0].shape.rank > 0:
            raise ValueError(
                'Only scalar actions are supported now, but action spec is: {}'
                .format(action_spec))

        spec = flat_action_spec[0]

        # TODO(b/119321125): Disable this once index_with_actions supports
        # negative-valued actions.
        if spec.minimum != 0:
            raise ValueError('Action specs should have minimum of 0, but saw: {0}'.format(spec))

        self._num_actions = spec.maximum - spec.minimum + 1
    
    def _get_target_updater(self, tau=1.0, period=1):
        """Performs a soft update of the target network parameters.

        For each weight w_s in the q network, and its corresponding
        weight w_t in the target_q_network, a soft update is:
        w_t = (1 - tau) * w_t + tau * w_s

        Args:
        tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
        period: Step interval at which the target network is updated.

        Returns:
        A callable that performs a soft update of the target network parameters.
        """
        with tf.name_scope('update_targets'):
            def update():
                return common.soft_variables_update(
                    self._q_network.variables,
                    self._target_q_network.variables,
                    tau,
                    tau_non_trainable=1.0)

            return common.Periodically(update, period, 'periodic_update_targets')
            
    # Use @common.function in graph mode or for speeding up.
    def _train(self, experience, weights):
        with tf.GradientTape() as tape:
            loss_info = self._loss(
                experience,
                td_errors_loss_fn=self._td_errors_loss_fn,
                gamma=self._gamma,
                reward_scale_factor=self._reward_scale_factor,
                weights=weights,
                training=True,
                retrace_lambda = self._retrace_lambda)
            
        
        tf.debugging.check_numerics(loss_info.loss, 'Loss is inf or nan')
        variables_to_train = self._q_network.trainable_weights
        non_trainable_weights = self._q_network.non_trainable_weights
        assert list(variables_to_train), "No variables in the agent's q_network."
        grads = tape.gradient(loss_info.loss, variables_to_train)
        # Tuple is used for py3, where zip is a generator producing values once.
        grads_and_vars = list(zip(grads, variables_to_train))
        if self._gradient_clipping is not None:
            grads_and_vars = eager_utils.clip_gradient_norms(grads_and_vars,
                                                        self._gradient_clipping)

        if self._summarize_grads_and_vars:
            grads_and_vars_with_non_trainable = (
            grads_and_vars + [(None, v) for v in non_trainable_weights])
            eager_utils.add_variables_summaries(grads_and_vars_with_non_trainable,self.train_step_counter)
            eager_utils.add_gradients_summaries(grads_and_vars,self.train_step_counter)
        
        self._optimizer.apply_gradients(grads_and_vars)
        self.train_step_counter.assign_add(1)

        self._update_target()

        return loss_info
    
    @tf.function
    def _loss(self, experience: Trajectory, td_errors_loss_fn=common.element_wise_huber_loss, gamma=1.0,
            reward_scale_factor=1.0, weights=None, training=False, retrace_lambda = 0.8):
        """Computes loss for DQN training.

        Args:
        experience: A batch of experience data in the form of a `Trajectory'. 
            The structure of `experience` must match that of
            `self.collect_policy.step_spec`.

            If a `Trajectory`, all tensors in `experience` must be shaped
            `[B, T, ...]` where `T` must be equal to `self.train_sequence_length`
            if that property is not `None`.
        td_errors_loss_fn: A function(td_targets, predictions) to compute the
            element wise loss.
        gamma: Discount for future rewards.
        reward_scale_factor: Multiplicative factor to scale rewards.
        weights: Optional scalar or elementwise (per-batch-entry) importance
            weights.  The output td_loss will be scaled by these weights, and
            the final scalar loss is the mean of these values.
        training: Whether this loss is being used for training.
        retrace_lambda: Look at init for more information

        Returns:
        loss: An instance of `DqnLossInfo`.
        Raises:
        ValueError:
            if the number of actions is greater than 1.
        """
        batch_size = tf.shape(experience.observation)[0]
        
        network_state = self.policy.get_initial_state(batch_size)
            
        # Extract trajectory components
        step_type = experience.step_type
        observation = experience.observation
        action = experience.action
        reward = experience.reward
        policy_info = experience.policy_info
        
        # Slice burn-in and n-step windows
        burn_in = self._burn_in
        n_step = self._n_step_update

         # Burn-in pass (warm up network state)
        if burn_in > 0:
            *_, network_state = self._q_network(
                observation[:, :burn_in],
                step_type=step_type[:, :burn_in],
                network_state=network_state,
                training=training)
            network_state = tf.nest.map_structure(tf.stop_gradient, network_state)
            
            
        """ Calculating q values """
        # Shapes [B,n+1,A], where A is the actio dim
        # slice off observations and step types
        observation = observation[:, burn_in: burn_in + n_step + 1] # [B, N+1, F]
        step_type = step_type[:, burn_in: burn_in + n_step + 1]     # [B, N+1]
        
        
        q_values, _= self._q_network(
            observation,
            step_type = step_type,
            network_state = network_state,
            training = training)
            
        
        target_q_values, _ = self._target_q_network(
            observation,
            step_type = step_type,
            network_state = network_state,
            training = False)
        
        # slice off rewards, discounts, and actions to match the q_values and target_q_values
        reward = reward[:, burn_in: burn_in + n_step] * reward_scale_factor # [B,N]
        action = action[:, burn_in: burn_in + n_step]  # [B, N]
        mu_log_probs = policy_info[:, burn_in : burn_in + n_step]  #[B,N]
        
        td_error, td_targets, q0a0 = self.compute_lambda_td_error(q_values, target_q_values,
                                            reward, step_type, action, mu_log_probs, gamma, n_step, retrace_lambda)
        
        # Compute loss
        td_loss = td_errors_loss_fn(td_targets, q0a0) #[B,N]
        
        # Aggregate across the elements of the batch and add regularization loss.
        # Note: We use an element wise loss above to ensure each element is always
        #   weighted by 1/N where N is the batch size, even when some of the
        #   weights are zero due to boundary transitions. Weighting by 1/K where K
        #   is the actual number of non-zero weight would artificially increase
        #   their contribution in the loss. Think about what would happen as
        #   the number of boundary samples increases.

 
        agg_loss = common.aggregate_losses(per_example_loss=td_loss,sample_weight=weights,regularization_loss=self._q_network.losses)
        total_loss = agg_loss.total_loss
            
        losses_dict = {'td_loss': agg_loss.weighted,'reg_loss': agg_loss.regularization,'total_loss': total_loss}

        common.summarize_scalar_dict(losses_dict,step=self.train_step_counter,name_scope='Losses/')

        if self._summarize_grads_and_vars:
            with tf.name_scope('Variables/'):
                for var in self._q_network.trainable_weights:
                    tf.compat.v2.summary.histogram(
                        name=var.name.replace(':', '_'),
                        data=var,
                        step=self.train_step_counter)

        if self._debug_summaries:
            diff_q_values = q0a0 - target_q_values
            common.generate_tensor_summaries('td_error', td_error, self.train_step_counter)
            common.generate_tensor_summaries('td_loss', td_loss,self.train_step_counter)
            common.generate_tensor_summaries('q0_values', q0a0,self.train_step_counter)
            common.generate_tensor_summaries('target_q_values', target_q_values,self.train_step_counter)
            common.generate_tensor_summaries('diff_q_values', diff_q_values,self.train_step_counter)

        return tf_agent.LossInfo(total_loss, DqnLossInfo(td_loss=td_loss,td_error=td_error))

    @tf.function
    def compute_lambda_td_error(
        self,
        q_values: tf.Tensor,             # shape [B, N+1, A]
        target_q_values: tf.Tensor,      # shape [B, N+1, A]
        rewards: tf.Tensor,              # shape [B, N]
        step_types: tf.Tensor,           # shape [B, N+1]
        actions: tf.Tensor,              # shape [B, N]
        mu_log_probs: tf.Tensor,         # shape [B, N]
        gamma: float,
        n_step: int,
        retrace_lambda: float
    ) :
        """
        Compute the Retrace(λ) TD-error for a batch of n-step transitions.

        Args:
            q_values:         Online Q network outputs over T = n_step+1 timesteps, shape [B, N+1, A].
            target_q_values:  Target Q network outputs over the same window, shape [B, N+1, A].
            rewards:          Collected rewards for n_step, shape [B, N].
            step_types:       StepType flags indicating episode boundaries, shape [B, N].
            actions:          Actions taken at each timestep, shape [B, N].
            mu_log_probs:     Log-probabilities under the behavior policy μ, shape [B, N].
            gamma:            Discount factor γ.
            n_step:           Number of steps N for multi-step returns.
            retrace_lambda:             Trace-decay parameter λ.

        Returns:
            td_err:   Tensor of per-batch TD-errors Δ_t = G_ret − Q(s₀,a₀), shape [B].
            G_ret:    Retrace return G^{Ret}_t for each batch entry, shape [B].
            q0a0:     Online Q(s₀,a₀) baseline for each batch entry, shape [B].
        """
        
        # 0) Build mask so that mask[b,t]==1 iff no boundary in [0..t], else 0.
        mask = tf.cast(step_types != StepType.LAST, tf.float32)  # [B, N+1]
        curr_q_val_mask = tf.math.cumprod(mask[:,:-1], axis=1)               # [B, N]
        next_q_val_mask = tf.math.cumprod(mask[:,1:], axis=1)               # [B, N]
        
        # 1) For each step t+i compute the next-step greedy action and its target Q:
        #    a*_t = argmax_a Q_online(s_t,a)
        best_a      = tf.argmax(q_values[:,1:,:], axis=2, output_type=tf.int32)                  # [B, N]
        q_star      = tf.gather(target_q_values[:,1:,:], best_a, axis=2, batch_dims=2)  # [B, N]
        
        
        # 2) Compute the one-step TD-errors δ:
        #    δ_t = r_t + γ * Q⋆_{t+1} - Q_online(s_t,a_t)
        q_ta   = tf.gather(q_values[:,:-1,:], actions, axis=2, batch_dims=2)  # [B,N]
        one_step_delta = (rewards + gamma*q_star*next_q_val_mask - q_ta)*curr_q_val_mask    # [B,N] 
        
        # 3) Compute importance-sampling ratios c_t = λ min(1, π(a_t|s_t)/μ(a_t|s_t))
        is_greedy = tf.equal(actions[:, 1:],  best_a[:,:-1])           # [B, N-1]
        ratio     = tf.where(is_greedy, 1.0/tf.exp(mu_log_probs[:, 1:]),tf.zeros_like(mu_log_probs[:, 1:]))
        c         = retrace_lambda*tf.minimum(1.0, ratio)       #  [B,N-1]
        c         = tf.concat([tf.ones_like(c[:,:1]), c], axis=1)       # [B, N]
        
        # 4) Build the Retrace weight sequence w_i = (γ)^i ∏_{j=1}^i c_{t+j}          
        prod_c        = tf.math.cumprod(c, axis=1)       # [B, N]
        weights       = self._gamma_powers * prod_c   # [B, N]
        
        # 5) Finally form the Retrace correction:
        #    G_ret = Q(s_0,a_0) + sum_{i=0..T-2} weights[:,i] * delta[:,i]
        q0a0   = tf.gather(q_values[:,0,:], actions[:,0], axis=1, batch_dims=1)         # [B]
        G_ret  = q0a0 + tf.reduce_sum(weights * one_step_delta, axis=1)                 # [B]
        td_err = G_ret - q0a0                                                           # [B]
        
        return td_err, G_ret, q0a0
    
    @tf.function
    def compute_legal_loss(
        self,
        legal_logit: tf.Tensor,  # [B, N, A]
        observation: tf.Tensor,   # [B, N, F]
        step_types: tf.Tensor,    # [B, N]
    ):
        if self._legality_one_hot_list is None:
            raise ValueError("Legality List required for legality computation")
        
        # Extract previous actions [B, N, 5]
        prev_actions = observation[:, :, 0 : self._num_actions] # slicing is exclusive
        
        # Convert to indices [B, N]
        action_idx = tf.argmax(prev_actions, axis=-1, output_type=tf.int32)
        
        # Gather legal masks [B, N, A]
        legal_mask = tf.gather(self._legality_one_hot_list, action_idx)
        
        # Create validity mask [B, N]
        valid_mask = tf.cast(step_types != StepType.LAST, tf.float32)
        valid_mask = tf.math.cumprod(valid_mask, axis=1)               # [B, N]
        
        # Compute per-step loss [B, N]
        per_step_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=legal_logit,
                labels=tf.cast(legal_mask, tf.float32)
            ),
            axis=-1
        )
        # Apply validity mask and normalize
        loss = tf.reduce_sum(per_step_loss * valid_mask) / tf.reduce_sum(valid_mask)
        return loss
            
            