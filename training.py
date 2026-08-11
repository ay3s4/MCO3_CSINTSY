import random
import time
from typing import Dict, List, Tuple
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################

N_ACTIONS = 4  # up, down, left, right
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GRID_SIZE = 8  # 10x10 grid

def decode_state(state: int) -> tuple:
    bot_x = (state // 1000) % 10
    bot_y = (state // 100) % 10
    cat_x = (state // 10) % 10
    cat_y = state % 10
    return bot_x, bot_y, cat_x, cat_y

def manhattan_distance(obs: int) -> int:
    bot_x, bot_y, cat_x, cat_y = decode_state(obs)
    return abs(bot_x - cat_x) + abs(bot_y - cat_y)


def choose_action(q_table, state: int, epsilon: float) -> int:
    """Epsilon-greedy policy: explore with probability epsilon, else exploit."""
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)
    return int(np.argmax(q_table[state]))

def decay_epsilon(epsilon: float, decay: float, epsilon_min: float) -> float:
    return max(epsilon * decay, epsilon_min)

PRIOR_SCALE = 0.01

def prior_q(state: int) -> np.ndarray:
    """Weak initial opinion for every state: prefer moves that close distance.
 
    Without this, a state training never visits keeps all-zero Q-values, so
    argmax always returns action 0 (walk into the top wall). That matters
    most for the 5 hidden cats, which are the states most likely to be
    unfamiliar at evaluation time. The scale is tiny on purpose so a single
    real training update overrides it.
    """
    bot_x, bot_y, cat_x, cat_y = decode_state(state)
    values = np.zeros(N_ACTIONS)
    for action, (dr, dc) in enumerate(MOVES):
        new_r = min(max(0, bot_x + dr), GRID_SIZE - 1)
        new_c = min(max(0, bot_y + dc), GRID_SIZE - 1)
        values[action] = -PRIOR_SCALE * (abs(new_r - cat_x) + abs(new_c - cat_y))
    return values


def compute_reward(
    prev_obs: int,
    next_obs: int,
    caught: bool,
    step_limit_reached: bool,
    *,
    gamma: float,
    step_penalty: float,
    distance_coef: float,
    catch_reward: float,
    trunc_penalty: float,
) -> float:
    """Shaped reward, since the env always returns 0.
    - catch_reward for catching the cat.
    - step_penalty on every step, to keep chases short (matters for the
      60-move eval cap).
    - trunc_penalty specifically for running out of moves without catching,
      distinct from an ordinary step, so the bot learns that stalling is
      worse than just being slow.
    - distance_coef * potential-based shaping (gamma*phi(s') - phi(s), with
      phi = -distance). This form doesn't change what the optimal policy is,
      it only speeds up learning, and it won't overwhelm the catch reward
      the way a flat distance bonus can for cats that punish naively closing
      in (e.g. Paotsin, Peekaboo).
    """
    if caught:
        return catch_reward
 
    prev_dist = manhattan_distance(prev_obs)
    next_dist = manhattan_distance(next_obs)
    shaping = (gamma * -next_dist) - (-prev_dist)
    reward = step_penalty + distance_coef * shaping
 
    if step_limit_reached:
        reward += trunc_penalty
 
    return reward

def greedy_score(env, q_table: Dict[int, np.ndarray], rollouts: int, max_steps: int) -> Tuple[int, float]:
    """Play `rollouts` greedy (no exploration) games. Returns (catches, mean steps)."""
    catches = 0
    total_steps = 0
    for _ in range(rollouts):
        state, _ = env.reset()
        for step in range(1, max_steps + 1):
            state, _, terminated, truncated, _ = env.step(int(np.argmax(q_table[state])))
            total_steps += 1
            if terminated:
                catches += 1
                break
            if truncated:
                break
    return catches, total_steps / rollouts



#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################
    
    
    gamma = 0.97              # discount factor
 
    alpha_start = 0.2         # learning rate, annealed: fast early learning,
    alpha_end = 0.05          # stable convergence later
 
    epsilon = 1.0              # start fully exploratory
    epsilon_min = 0.05         # keep a little exploration throughout
    epsilon_decay = (epsilon_min / epsilon) ** (1.0 / episodes)  # reaches
                                # epsilon_min by the final episode
 
    # Matches the 60-move cap used during evaluation, so training experience
    # mirrors what the bot actually faces, and no single bad episode can blow
    # the 20s training time budget.
    max_steps_per_episode = 60
 
    # Reward shaping constants
    catch_reward = 100.0
    step_penalty = -1.0
    distance_coef = 0.5
    trunc_penalty = -5.0
 
    # Start from a weak "move toward the cat" prior instead of all zeros, so
    # states training never visits still produce sensible moves. See prior_q.
    for state in q_table:
        q_table[state] = prior_q(state)
 
    # Snapshot + best-of-N selection. A run can be catching cats reliably at
    # episode 3000 and end up worse by episode 5000, since epsilon-greedy
    # exploration keeps perturbing the policy right up to the last episode.
    # Keeping periodic snapshots and picking whichever plays best avoids
    # handing back a policy that got unlucky right at the end. This only
    # spends extra environment steps on evaluation games, not extra training
    # episodes, so the 5000-episode budget is untouched.
    snapshot_every = 1000
    snapshot_rollouts = 12
    snapshots: List[Dict[int, np.ndarray]] = []
    eval_env = make_env(cat_type=cat_name)



    
    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 
               
        obs, _ = env.reset()
        alpha = alpha_start + (alpha_end - alpha_start) * (ep - 1) / episodes
        done = False
        step_count = 0
 
        while not done:
            #Explore or exploit.
            action = choose_action(q_table, obs, epsilon)
 
            #Take the action, observe the next state.
            next_obs, _, terminated, truncated, _ = env.step(action)
            step_count += 1
            step_limit_reached = step_count >= max_steps_per_episode
            done = terminated or truncated or step_limit_reached
 
            #Compute reward manually.
            reward = compute_reward(
                obs,
                next_obs,
                terminated,
                step_limit_reached,
                gamma=gamma,
                step_penalty=step_penalty,
                distance_coef=distance_coef,
                catch_reward=catch_reward,
                trunc_penalty=trunc_penalty,
            )
 
            #Update Q-table
            best_next = 0.0 if done else float(np.max(q_table[next_obs]))
            q_table[obs][action] += alpha * (reward + gamma * best_next - q_table[obs][action])
 
            obs = next_obs
 
        #Decay exploration after each episode.
        epsilon = decay_epsilon(epsilon, epsilon_decay, epsilon_min)
 
        if ep % snapshot_every == 0:
            snapshots.append({s: v.copy() for s, v in q_table.items()})
 
        #On the last episode, keep whichever snapshot plays best greedily,
        #instead of automatically returning the final table
        if ep == episodes:
            best_table, best_key = q_table, None
            for candidate in snapshots:
                catches, mean_steps = greedy_score(eval_env, candidate, snapshot_rollouts, max_steps_per_episode)
                key = (catches, -mean_steps)
                if best_key is None or key > best_key:
                    best_table, best_key = candidate, key
            eval_env.close()
            q_table = best_table

        
        
        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table