from tf_agents.agents.dqn.dqn_agent import DqnAgent, DqnLossInfo
import tensorflow as tf
from tf_agents.agents import tf_agent
from tf_agents.utils import nest_utils
from tf_agents.utils import common
from tf_agents.trajectories import Trajectory, StepType

class DdqnAgent(DqnAgent):
    
    def _loss(self, experience: Trajectory, td_errors_loss_fn=None, gamma=1.0, reward_scale_factor=1.0, weights=None, training=False):
        """Computes loss for DQN training.

        Args:
        experience: A batch of experience data in the form of a `Trajectory`.
        td_errors_loss_fn: A function(td_targets, predictions) to compute the
        element wise loss.
        gamma: Discount for future rewards.
        reward_scale_factor: Multiplicative factor to scale rewards.
        weights: Optional scalar or elementwise (per-batch-entry) importance
        weights.  The output td_loss will be scaled by these weights, and the
        final scalar loss is the mean of these values.
        training: Whether this loss is being used for training.

        Returns:
        loss: An instance of `DqnLossInfo`.
        Raises:
        ValueError:
        if the number of actions is greater than 1.
        """
        N = self._n_step_update

        # N is an int or tf.Tensor scalar
        observation = tf.nest.map_structure(lambda x: x[:, 0:N+1], experience.observation) # [B, N+1, F], sliced from indx 0,1,...,N-1,N
        step_type = experience.step_type[:, 0:N+1]         # [B, N+1]
        action = experience.action[:, 0]                  # [B]
        reward = experience.reward[:, 0:N]                # [B, N]

        valid_mask = tf.cast(step_type != StepType.LAST, tf.float32)
        valid_mask = tf.math.cumprod(valid_mask, axis = -1) # [B, N+1]

        q0,_ = self._q_network(tf.nest.map_structure(lambda x: x[:,0], observation), step_type[:,0], network_state = (), training = training)    # [B,A]
        q0a0 = tf.gather(q0, action, batch_dims=1, axis = 1)        # [B]

        qn,_ = self._q_network(tf.nest.map_structure(lambda x: x[:,N], observation), step_type[:,N], network_state = (), training = training)  # [B,A]
        best_ind = tf.math.argmax(qn, axis = -1, output_type=tf.int32)              # [B]
        tn,_ = self._target_q_network(tf.nest.map_structure(lambda x: x[:,N], observation), step_type[:,N], network_state = (), training = False)  # [B,A]
        tn_astar = tf.gather(tn, best_ind, batch_dims=1, axis= 1)

        gammas = gamma ** tf.range(N, dtype=tf.float32)
        cumul_reward = tf.math.reduce_sum( gammas * reward * valid_mask[:, 0:N], axis = -1) * reward_scale_factor # valid indexing goes from 0 to N-1
        bootstrap = tn_astar * valid_mask[:, N] * gamma**N         # indexes the Nth element
        td_target = cumul_reward + bootstrap

        td_error = td_target - q0a0
        td_loss = td_errors_loss_fn(td_target, q0a0)
        
        # Aggregate across the elements of the batch and add regularization loss.
        # Note: We use an element wise loss above to ensure each element is always
        #   weighted by 1/N where N is the batch size, even when some of the
        #   weights are zero due to boundary transitions. Weighting by 1/K where K
        #   is the actual number of non-zero weight would artificially increase
        #   their contribution in the loss. Think about what would happen as
        #   the number of boundary samples increases.

        agg_loss = common.aggregate_losses(
        per_example_loss=td_loss,
        sample_weight=weights,
        regularization_loss=self._q_network.losses,
        )
        total_loss = agg_loss.total_loss

        losses_dict = {
        'td_loss': agg_loss.weighted,
        'reg_loss': agg_loss.regularization,
        'total_loss': total_loss,
        }

        common.summarize_scalar_dict(
        losses_dict, step=self.train_step_counter, name_scope='Losses/'
        )

        if self._summarize_grads_and_vars:
            with tf.name_scope('Variables/'):
                for var in self._q_network.trainable_weights:
                    tf.compat.v2.summary.histogram(
                        name=var.name.replace(':', '_'),
                        data=var,
                        step=self.train_step_counter,
                        )

        if self._debug_summaries:
            diff_q_values = q0a0 - tn_astar
            common.generate_tensor_summaries( 'td_error', td_error, self.train_step_counter)
            common.generate_tensor_summaries( 'td_loss', td_loss, self.train_step_counter)
            common.generate_tensor_summaries('q_values', q0a0, self.train_step_counter)
            common.generate_tensor_summaries('next_q_values', tn_astar, self.train_step_counter)
            common.generate_tensor_summaries('diff_q_values', diff_q_values, self.train_step_counter)

        return tf_agent.LossInfo(
        total_loss, DqnLossInfo(td_loss=td_loss, td_error=td_error)
        )
