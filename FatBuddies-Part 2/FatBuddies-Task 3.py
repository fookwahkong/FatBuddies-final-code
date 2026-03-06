import importlib.util
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from GridWorld import Action, GridWorld, State


def load_task2_module():
    task2_path = Path(__file__).with_name("FatBuddies-Task 2.py")
    spec = importlib.util.spec_from_file_location("fatbuddies_task2", task2_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Task 2 module from {task2_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TASK2 = load_task2_module()
MonteCarloAgent = TASK2.MonteCarloAgent
compare_policies = TASK2.compare_policies
compare_optimal_actions = TASK2.compare_optimal_actions


class QLearningAgent:
    def __init__(
        self,
        environment: GridWorld,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        episodes: int = 50000,
        max_steps_per_episode: int = 500,
        seed: int = 7,
    ) -> None:
        self.environment = environment
        self.epsilon = epsilon
        self.alpha = alpha
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.random = random.Random(seed)

        self.q_values: Dict[Tuple[State, Action], float] = {
            (state, action): 0.0
            for state in self.environment.states
            if not self.environment.is_terminal(state)
            for action in self.environment.actions
        }

    def epsilon_greedy_action(self, state: State) -> Action:
        if self.random.random() < self.epsilon:
            return self.random.choice(self.environment.actions)

        return max(
            self.environment.actions,
            key=lambda action: self.q_values[(state, action)],
        )

    def greedy_policy(self) -> Dict[State, Optional[Action]]:
        policy: Dict[State, Optional[Action]] = {}

        for state in self.environment.states:
            if self.environment.is_terminal(state):
                policy[state] = None
                continue

            policy[state] = max(
                self.environment.actions,
                key=lambda action: self.q_values[(state, action)],
            )

        return policy

    def state_values_from_q(self) -> Dict[State, float]:
        values: Dict[State, float] = {}

        for state in self.environment.states:
            if self.environment.is_terminal(state):
                values[state] = 0.0
                continue

            values[state] = max(
                self.q_values[(state, action)]
                for action in self.environment.actions
            )

        return values

    def train(self, checkpoints: List[int]) -> Tuple[List[Dict[str, float]], List[float]]:
        checkpoint_set = set(checkpoints)
        checkpoint_metrics: List[Dict[str, float]] = []
        episode_rewards: List[float] = []

        for episode_index in range(1, self.episodes + 1):
            state = self.environment.start
            total_reward = 0.0

            for _ in range(self.max_steps_per_episode):
                if self.environment.is_terminal(state):
                    break

                action = self.epsilon_greedy_action(state)
                next_state = self.environment.move(state, action)
                reward = self.environment.get_reward(next_state)
                total_reward += reward

                old_q = self.q_values[(state, action)]
                if self.environment.is_terminal(next_state):
                    target = reward
                else:
                    target = reward + self.environment.gamma * max(
                        self.q_values[(next_state, next_action)]
                        for next_action in self.environment.actions
                    )

                self.q_values[(state, action)] = old_q + self.alpha * (target - old_q)
                state = next_state

                if self.environment.is_terminal(state):
                    break

            episode_rewards.append(total_reward)

            if episode_index in checkpoint_set:
                policy = self.greedy_policy()
                checkpoint_metrics.append(
                    build_metrics(
                        method_name="Q-learning",
                        episode_index=episode_index,
                        recent_rewards=episode_rewards,
                        policy=policy,
                        optimal_policy=OPTIMAL_POLICY,
                        optimal_values=OPTIMAL_VALUES,
                        environment=self.environment,
                    )
                )

        return checkpoint_metrics, episode_rewards


def train_monte_carlo_with_checkpoints(
    environment: GridWorld,
    checkpoints: List[int],
    epsilon: float = 0.1,
    episodes: int = 50000,
    max_steps_per_episode: int = 500,
    seed: int = 42,
) -> Tuple[MonteCarloAgent, List[Dict[str, float]], List[float]]:
    agent = MonteCarloAgent(
        environment=environment,
        epsilon=epsilon,
        episodes=episodes,
        max_steps_per_episode=max_steps_per_episode,
        seed=seed,
    )
    checkpoint_set = set(checkpoints)
    checkpoint_metrics: List[Dict[str, float]] = []
    episode_rewards: List[float] = []

    for episode_index in range(1, episodes + 1):
        episode = agent.generate_episode()
        episode_rewards.append(sum(reward for _, _, reward in episode))

        visited_state_actions = set()
        returns_so_far = 0.0

        for state, action, reward in reversed(episode):
            returns_so_far = environment.gamma * returns_so_far + reward
            state_action = (state, action)

            if state_action in visited_state_actions:
                continue

            visited_state_actions.add(state_action)
            agent.returns_sum[state_action] += returns_so_far
            agent.returns_count[state_action] += 1
            agent.q_values[state_action] = (
                agent.returns_sum[state_action] / agent.returns_count[state_action]
            )

        if episode_index in checkpoint_set:
            policy = agent.greedy_policy()
            checkpoint_metrics.append(
                build_metrics(
                    method_name="Monte Carlo",
                    episode_index=episode_index,
                    recent_rewards=episode_rewards,
                    policy=policy,
                    optimal_policy=OPTIMAL_POLICY,
                    optimal_values=OPTIMAL_VALUES,
                    environment=environment,
                )
            )

    return agent, checkpoint_metrics, episode_rewards


def build_metrics(
    method_name: str,
    episode_index: int,
    recent_rewards: List[float],
    policy: Dict[State, Optional[Action]],
    optimal_policy: Dict[State, Optional[Action]],
    optimal_values: Dict[State, float],
    environment: GridWorld,
) -> Dict[str, float]:
    exact_matches, total_states, _ = compare_policies(policy, optimal_policy, environment)
    optimal_action_matches, _, suboptimal_states = compare_optimal_actions(
        policy,
        optimal_values,
        environment,
    )
    policy_values, _ = environment.policy_evaluation(policy)
    max_value_gap = max(
        abs(policy_values[state] - optimal_values[state])
        for state in environment.states
    )
    recent_window = recent_rewards[-100:] if len(recent_rewards) >= 100 else recent_rewards
    average_recent_reward = sum(recent_window) / len(recent_window)

    return {
        "episode": float(episode_index),
        "exact_match_rate": exact_matches / total_states,
        "optimal_action_rate": optimal_action_matches / total_states,
        "max_value_gap": max_value_gap,
        "average_recent_reward": average_recent_reward,
        "suboptimal_count": float(len(suboptimal_states)),
        "start_state_value": policy_values[environment.start],
        "method": method_name,
    }


def print_checkpoint_table(metrics: List[Dict[str, float]]) -> None:
    print(
        f"{'Method':<13} {'Episodes':>8} {'AvgReward100':>12} "
        f"{'ExactMatch':>12} {'OptimalActs':>12} {'ValueGap':>12}"
    )
    for item in metrics:
        print(
            f"{item['method']:<13} {int(item['episode']):>8} "
            f"{item['average_recent_reward']:>12.3f} "
            f"{item['exact_match_rate']:>11.2%} "
            f"{item['optimal_action_rate']:>11.2%} "
            f"{item['max_value_gap']:>12.6f}"
        )


def first_optimal_checkpoint(metrics: List[Dict[str, float]]) -> Optional[int]:
    for item in metrics:
        if item["optimal_action_rate"] == 1.0 and item["suboptimal_count"] == 0.0:
            return int(item["episode"])
    return None


def summarize_policy_comparison(
    learned_policy: Dict[State, Optional[Action]],
    reference_policy: Dict[State, Optional[Action]],
    environment: GridWorld,
) -> Tuple[int, int, List[State]]:
    return compare_policies(learned_policy, reference_policy, environment)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    environment = GridWorld(gamma=0.9)
    checkpoints = [100, 500, 1000, 5000, 10000, 25000, 50000]

    monte_carlo_agent, mc_metrics, _ = train_monte_carlo_with_checkpoints(
        environment=environment,
        checkpoints=checkpoints,
        epsilon=0.1,
        episodes=50000,
        max_steps_per_episode=500,
        seed=42,
    )

    q_learning_agent = QLearningAgent(
        environment=environment,
        epsilon=0.1,
        alpha=0.1,
        episodes=50000,
        max_steps_per_episode=500,
        seed=7,
    )
    q_metrics, _ = q_learning_agent.train(checkpoints)

    monte_carlo_policy = monte_carlo_agent.greedy_policy()
    q_learning_policy = q_learning_agent.greedy_policy()
    q_learning_values = q_learning_agent.state_values_from_q()
    q_learning_policy_values, _ = environment.policy_evaluation(q_learning_policy)

    q_vs_optimal_matches, q_total_states, q_vs_optimal_mismatches = summarize_policy_comparison(
        q_learning_policy,
        OPTIMAL_POLICY,
        environment,
    )
    q_vs_mc_matches, _, q_vs_mc_mismatches = summarize_policy_comparison(
        q_learning_policy,
        monte_carlo_policy,
        environment,
    )
    q_optimal_action_matches, _, q_suboptimal_states = compare_optimal_actions(
        q_learning_policy,
        OPTIMAL_VALUES,
        environment,
    )
    q_value_gap = max(
        abs(q_learning_policy_values[state] - OPTIMAL_VALUES[state])
        for state in environment.states
    )

    print("Grid World Task 3")
    print(f"Start state: {environment.start}")
    print(f"Goal state: {environment.goal}")
    print(f"Roadblocks: {sorted(environment.roadblocks)}")
    print(f"Discount factor: {environment.gamma}")
    print("Epsilon: 0.1")
    print("Learning rate alpha: 0.1")
    print("Training episodes per method: 50000")

    print("\nOptimal policy from Task 1")
    print(environment.format_policy(OPTIMAL_POLICY))

    print("\nMonte Carlo policy from Task 2")
    print(environment.format_policy(monte_carlo_policy))

    print("\nLearned Q-learning policy")
    print(environment.format_policy(q_learning_policy))

    print("\nApproximate state values from Q-learning")
    print(environment.format_values(q_learning_values))

    print("\nExact value function of the Q-learning policy")
    print(environment.format_values(q_learning_policy_values))

    print("\nPolicy comparison")
    print(
        f"Q-learning vs Task 1 exact match: "
        f"{q_vs_optimal_matches}/{q_total_states} ({q_vs_optimal_matches / q_total_states:.2%})"
    )
    print(
        f"Q-learning actions that are still optimal: "
        f"{q_optimal_action_matches}/{q_total_states} ({q_optimal_action_matches / q_total_states:.2%})"
    )
    print(
        f"Q-learning vs Monte Carlo exact match: "
        f"{q_vs_mc_matches}/{q_total_states} ({q_vs_mc_matches / q_total_states:.2%})"
    )
    print(f"States differing from Task 1 policy: {q_vs_optimal_mismatches if q_vs_optimal_mismatches else 'None'}")
    print(f"States differing from Monte Carlo policy: {q_vs_mc_mismatches if q_vs_mc_mismatches else 'None'}")
    print(f"States with genuinely suboptimal Q-learning action: {q_suboptimal_states if q_suboptimal_states else 'None'}")
    print(f"Maximum value gap vs optimal policy: {q_value_gap:.12f}")

    print("\nConvergence checkpoints")
    print_checkpoint_table(mc_metrics + q_metrics)

    mc_first_optimal = first_optimal_checkpoint(mc_metrics)
    q_first_optimal = first_optimal_checkpoint(q_metrics)

    print("\nAnalysis")
    print(
        f"Monte Carlo reached an optimal-action policy by episode: "
        f"{mc_first_optimal if mc_first_optimal is not None else 'Not reached in checkpoints'}"
    )
    print(
        f"Q-learning reached an optimal-action policy by episode: "
        f"{q_first_optimal if q_first_optimal is not None else 'Not reached in checkpoints'}"
    )

    if mc_first_optimal is not None and q_first_optimal is not None:
        if q_first_optimal < mc_first_optimal:
            print("Q-learning converged faster in this environment because it updates after every step.")
        elif q_first_optimal > mc_first_optimal:
            print("Monte Carlo converged faster in this environment based on the checkpoint results.")
        else:
            print("Monte Carlo and Q-learning reached the optimal-action policy at the same checkpoint.")

    print(
        "Q-learning is more sample-efficient here because it bootstraps from the next-state estimate, "
        "while Monte Carlo waits for the full episode return before updating."
    )


ENVIRONMENT = GridWorld(gamma=0.9)
OPTIMAL_VALUES, OPTIMAL_POLICY, _ = ENVIRONMENT.value_iteration()


if __name__ == "__main__":
    main()
