import random
import sys
from typing import Dict, List, Optional, Tuple

from GridWorld import Action, GridWorld, State


class MonteCarloAgent:
    def __init__(
        self,
        environment: GridWorld,
        epsilon: float = 0.1,
        episodes: int = 50000,                  #number of training episodes
        max_steps_per_episode: int = 200,       #set maximum to avoid infinte wandering 
        seed: int = 42,
    ) -> None:
        self.environment = environment
        self.epsilon = epsilon
        self.episodes = episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.random = random.Random(seed)

        self.q_values: Dict[Tuple[State, Action], float] = {
            (state, action): 0.0
            for state in self.environment.states
            if not self.environment.is_terminal(state)         #exclude terminal states
            for action in self.environment.actions             #loop through all four actions for each state
        }
        self.returns_sum: Dict[Tuple[State, Action], float] = {
            key: 0.0 for key in self.q_values
        }
        self.returns_count: Dict[Tuple[State, Action], int] = {
            key: 0 for key in self.q_values
        }

    def epsilon_greedy_action(self, state: State) -> Action:
        if self.random.random() < self.epsilon:                     #agent exploring random action
            return self.random.choice(self.environment.actions)     

        return max(                                                 #else, choose the best current action (max q values)
            self.environment.actions,
            key=lambda action: self.q_values[(state, action)],
        )

    def generate_episode(self) -> List[Tuple[State, Action, float]]:
        episode: List[Tuple[State, Action, float]] = []
        state = self.environment.start

        for _ in range(self.max_steps_per_episode):               
            if self.environment.is_terminal(state):               #reach end state then stop
                break

            action = self.epsilon_greedy_action(state)
            next_state = self.environment.move(state, action)
            reward = self.environment.get_reward(next_state)
            episode.append((state, action, reward))
            state = next_state

            if self.environment.is_terminal(state):
                break

        return episode

    def train(self) -> Dict[Tuple[State, Action], float]:
        '''
        Monte Carlo training
        1. Generates many episodes
        2. For each episode
        3. Compute returns backwards
        4. For each first occurrence of (state, action)
        5. Record the return
        6. Update Q-value as the average return
        '''
        for _ in range(self.episodes):
            episode = self.generate_episode()
            visited_state_actions = set()
            returns_so_far = 0.0

            for state, action, reward in reversed(episode):
                returns_so_far = self.environment.gamma * returns_so_far + reward         #discounted return = gamma * G + reward
                state_action = (state, action)

                #First visit MC
                if state_action in visited_state_actions:
                    continue

                visited_state_actions.add(state_action)
                self.returns_sum[state_action] += returns_so_far
                self.returns_count[state_action] += 1
                self.q_values[state_action] = (
                    self.returns_sum[state_action] / self.returns_count[state_action]
                )

        return self.q_values

    def greedy_policy(self) -> Dict[State, Optional[Action]]:
        '''
        Extract the learned greedy policy
        '''
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
        '''
        Convert Q-values to state values
        '''
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


def compare_policies(
    learned_policy: Dict[State, Optional[Action]],
    reference_policy: Dict[State, Optional[Action]],
    environment: GridWorld,
) -> Tuple[int, int, List[State]]:
    matches = 0
    mismatches: List[State] = []

    for state in environment.states:
        if environment.is_terminal(state):
            continue

        if learned_policy[state] == reference_policy[state]:
            matches += 1
        else:
            mismatches.append(state)

    return matches, len(environment.states) - 1, mismatches


def compare_optimal_actions(
    learned_policy: Dict[State, Optional[Action]],
    optimal_values: Dict[State, float],
    environment: GridWorld,
    tolerance: float = 1e-9,
) -> Tuple[int, int, List[State]]:
    optimal_action_matches = 0
    suboptimal_states: List[State] = []

    for state in environment.states:
        if environment.is_terminal(state):
            continue

        learned_action = learned_policy[state]
        assert learned_action is not None

        learned_q = environment.compute_q_value(state, learned_action, optimal_values)
        optimal_q = max(
            environment.compute_q_value(state, action, optimal_values)
            for action in environment.actions
        )

        if abs(learned_q - optimal_q) <= tolerance:
            optimal_action_matches += 1
        else:
            suboptimal_states.append(state)

    return optimal_action_matches, len(environment.states) - 1, suboptimal_states


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    environment = GridWorld(gamma=0.9)
    optimal_values, optimal_policy, _ = environment.value_iteration()

    agent = MonteCarloAgent(
        environment=environment,
        epsilon=0.1,
        episodes=50000,
        max_steps_per_episode=500,
        seed=42,
    )
    agent.train()

    learned_policy = agent.greedy_policy()
    learned_values = agent.state_values_from_q()
    learned_policy_values, _ = environment.policy_evaluation(learned_policy)

    matches, total_states, mismatches = compare_policies(
        learned_policy,
        optimal_policy,
        environment,
    )
    optimal_action_matches, _, suboptimal_states = compare_optimal_actions(
        learned_policy,
        optimal_values,
        environment,
    )
    max_value_gap = max(
        abs(learned_policy_values[state] - optimal_values[state])
        for state in environment.states
    )

    print("Grid World Task 2")
    print(f"Start state: {environment.start}")
    print(f"Goal state: {environment.goal}")
    print(f"Roadblocks: {sorted(environment.roadblocks)}")
    print(f"Discount factor: {environment.gamma}")
    print(f"Epsilon: {agent.epsilon}")
    print(f"Training episodes: {agent.episodes}")

    print("\nOptimal policy from Task 1")
    print(environment.format_policy(optimal_policy))

    print("\nApproximate state values from Monte Carlo Q estimates")
    print(environment.format_values(learned_values))

    print("\nLearned policy from Monte Carlo control")
    print(environment.format_policy(learned_policy))

    print("\nExact value function of the learned policy")
    print(environment.format_values(learned_policy_values))

    print("\nComparison with Task 1 optimal policy")
    print(f"Exact match with Task 1 policy: {matches}/{total_states} ({matches / total_states:.2%})")
    print(
        f"States where learned action is still optimal: "
        f"{optimal_action_matches}/{total_states} ({optimal_action_matches / total_states:.2%})"
    )
    print(f"States with different action from Task 1 policy: {mismatches if mismatches else 'None'}")
    print(f"States with genuinely suboptimal action: {suboptimal_states if suboptimal_states else 'None'}")
    print(f"Maximum value gap vs optimal policy: {max_value_gap:.12f}")

    start_state_value = learned_policy_values[environment.start]
    optimal_start_value = optimal_values[environment.start]
    print(f"Learned-policy start-state value: {start_state_value:.3f}")
    print(f"Optimal start-state value: {optimal_start_value:.3f}")


if __name__ == "__main__":
    main()
    
