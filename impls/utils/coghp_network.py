from typing import Sequence, Optional
import distrax
import jax
import jax.numpy as jnp
import flax.linen as nn

from utils.networks import default_init, MLP, LengthNormalize


class MixerBlock(nn.Module):
    num_tokens: int
    embed_dim: int
    hidden_dim_tokens: int
    hidden_dim_channels: int
    init_scale: float = 1e-2

    decay_alpha: float = 0.9
    causal: bool = True

    def setup(self):
        self.token_dense1 = nn.Dense(self.hidden_dim_tokens, kernel_init=default_init())
        self.token_dense2 = nn.Dense(self.num_tokens, kernel_init=default_init())
        self.channel_dense1 = nn.Dense(self.hidden_dim_channels, kernel_init=default_init())
        self.channel_dense2 = nn.Dense(self.embed_dim, kernel_init=default_init())

        # Initialize learnable token-mixing weight matrix.
        self.tm_weights = self.param(
            'tm_weights',
            nn.initializers.normal(stddev=0.02),
            (self.num_tokens, self.num_tokens)
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, num_tokens, embed_dim)
        
        # Transpose for token mixing across the token dimension.
        y = jnp.transpose(x, (0, 2, 1))
        y = self.token_dense1(y)
        y = nn.gelu(y)
        y = self.token_dense2(y)
        y = jnp.transpose(y, (0, 2, 1))

        tm_weights = jnp.tril(self.tm_weights) if self.causal else self.tm_weights
        y = jnp.einsum('btd,ts->bsd', y, tm_weights)

        x = x + y  # residual connection

        # Channel mixing within each token.
        z = self.channel_dense1(x)
        z = nn.gelu(z)
        z = self.channel_dense2(z)

        output = x + z

        return output
    
class HierarchicalPolicyNetwork(nn.Module):
    """
    Attributes:
        num_tokens: Number of state tokens (sequence length).
        state_dim: Dimension of state features.
        num_action_dims: Total number of action dimensions.
        joint_embed_dim: Joint embedding dimension for mapping actions.
        num_mixer_blocks: Number of MixerBlock layers.
        mixer_token_hidden: Hidden dimension for token mixing.
        mixer_channel_hidden: Hidden dimension for channel mixing.
    """
    num_tokens: int
    state_dim: int
    num_action_dims: int
    joint_embed_dim: int = 128
    num_mixer_blocks: int = 2
    mixer_token_hidden: int = 64
    mixer_channel_hidden: int = 64
    gc_encoder: nn.Module = None
    layer_norm: bool = True
    final_fc_init_scale: float = 1e-2

    high_actor_head: nn.Module = None
    low_actor_head: nn.Module = None
    enc_hidden: Sequence[int] = (128, 128)

    num_subgoals: int = 1
    causal_mixer: bool = True
    action_use_full_subgoal_chain: bool = True
    share_mixer_weights: bool = False
    separate_action_mixer: bool = False

    def setup(self):
        # Parameter for previous token embeddings:
        self.prev_tokens = self.param("prev_tokens",
                                      nn.initializers.normal(stddev=0.1),
                                      (1, self.num_subgoals + 1, self.state_dim))

        if self.separate_action_mixer and self.share_mixer_weights:
            raise ValueError(
                'separate_action_mixer and share_mixer_weights cannot both be enabled.'
            )

        mixer_kwargs = dict(
            num_tokens=self.num_subgoals + 3,
            embed_dim=self.state_dim,
            hidden_dim_tokens=self.mixer_token_hidden,
            hidden_dim_channels=self.mixer_channel_hidden,
            causal=self.causal_mixer,
        )

        if self.separate_action_mixer:
            # Preserve the original CoGHP sharing across all subgoal prediction
            # steps, while giving primitive-action prediction its own mixer.
            self.subgoal_mixer_blocks = [
                MixerBlock(**mixer_kwargs) for _ in range(self.num_mixer_blocks)
            ]
            self.action_mixer_blocks = [
                MixerBlock(**mixer_kwargs) for _ in range(self.num_mixer_blocks)
            ]
        elif self.share_mixer_weights:
            self.shared_mixer_block = MixerBlock(
                **mixer_kwargs,
            )
        else:
            self.mixer_blocks = [
                MixerBlock(**mixer_kwargs)
                for _ in range(self.num_mixer_blocks)
            ]
        
        feature_embed = [MLP(hidden_dims=(*self.enc_hidden, self.state_dim), activate_final=False, layer_norm=True)]
        feature_embed.append(LengthNormalize())
        self.feature_embed = nn.Sequential(feature_embed)
        
    def __call__(self,
                 observations: jnp.ndarray,
                 goals: jnp.ndarray,
                 seed: int = None,
                 subgoal_reps: Optional[jnp.ndarray] = None,
                 action_seq: Optional[jnp.ndarray] = None,
                 temperature: float = 1.0):
        
        high_seed, low_seed = jax.random.split(seed)

        observations = jnp.expand_dims(observations, axis=1) # (B, 1, state_dim)
        if goals is not None:
            goals = jnp.expand_dims(goals, axis=1) # (B, 1, state_dim)
                
        if self.gc_encoder is not None:
            features = self.gc_encoder(observations, goals, goal_encoded=False, listwise=True)
            obs_feature = self.feature_embed(features[0])
            goal_feature = features[1]
            features = jnp.concatenate([obs_feature, goal_feature], axis=1)
        else:
            features = [self.feature_embed(observations)]
            if goals is not None:
                features.append(self.feature_embed(goals))            
            features = jnp.concatenate(features, axis=1)
        
        B, T, _ = features.shape
        
        # Repeat the prev_tokens.      
        predicted_subgoals = jnp.zeros((B, self.num_subgoals, self.joint_embed_dim), dtype=jnp.float32)  
        prev_embed_tokens = jnp.tile(self.prev_tokens, (B, 1, 1))

        high_dist_list = []
        for token_dim in range(self.num_subgoals + 1):
            if token_dim == 0:
                prev_embeds = prev_embed_tokens
            
            else:
                if subgoal_reps is not None:
                    history_subgoals = subgoal_reps[:, :token_dim, :]
                else:
                    history_subgoals = predicted_subgoals[:, :token_dim, :]

                if token_dim == self.num_subgoals and not self.action_use_full_subgoal_chain:
                    history_subgoals = prev_embed_tokens[:, :token_dim, :].at[:, token_dim - 1, :].set(
                        history_subgoals[:, token_dim - 1, :]
                    )

                prev_embeds = jnp.concatenate([history_subgoals, prev_embed_tokens[:, token_dim:, :]], axis=1)
            
            x = jnp.concatenate([features, prev_embeds], axis=1)

            target_dim = features.shape[1] + token_dim + 1

            # Apply Mixer blocks.
            if self.separate_action_mixer:
                if token_dim < self.num_subgoals:
                    mixer_blocks = self.subgoal_mixer_blocks
                else:
                    mixer_blocks = self.action_mixer_blocks
                for mixer_block in mixer_blocks:
                    x = mixer_block(x)
            elif self.share_mixer_weights:
                for _ in range(self.num_mixer_blocks):
                    x = self.shared_mixer_block(x)
            else:
                for mixer_block in self.mixer_blocks:
                    x = mixer_block(x)
            
            target_token = x[:, target_dim-1, :]

            if token_dim < self.num_subgoals:
                high_dist = self.high_actor_head(target_token, temperature=temperature)
                high_dist_list.append(high_dist)
                goal_reps = high_dist.sample(seed=high_seed)

                predicted_subgoals = predicted_subgoals.at[:, token_dim, :].set(goal_reps)

            else:
                low_dist = self.low_actor_head(target_token, temperature=temperature)  #(B, 1)
                predicted_actions = low_dist.sample(seed=low_seed)  # (B, num_action_dims)
                
        return high_dist_list, low_dist, predicted_actions
