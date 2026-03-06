from typing import Dict, List, Optional, Tuple

State = Tuple[int, int]
Action = str


class GridWorld:
    def __init__(
        self,
        width: int = 5,
        height: int = 5,
        start: State = (0, 0),
        goal: State = (4, 4),
        roadblocks: Optional[List[State]] = None,
        gamma: float = 0.9,
        step_reward: float = -1.0,
        goal_reward: float = 10.0,
    ) -> None:
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.roadblocks = set(roadblocks or [(1, 2), (3, 2)])
        self.gamma = gamma
        self.step_reward = step_reward
        self.goal_reward = goal_reward

        self.actions: List[Action] = ["U", "D", "L", "R"]
        self.action_delta = {
            "U": (0, 1),
            "D": (0, -1),
            "L": (-1, 0),
            "R": (1, 0),
        }

        self.states = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in self.roadblocks
        ]

    def in_bounds(self, state: State) -> bool:
        '''
        Check whether a state is inside the grid
        '''
        x, y = state
        return 0 <= x < self.width and 0 <= y < self.height

    def is_terminal(self, state: State) -> bool:
        '''
        Return True if reach terminal state
        '''
        return state == self.goal

    def move(self, state: State, action: Action) -> State:
        '''
        Move from state to state
        '''
        if self.is_terminal(state):
            return state

        dx, dy = self.action_delta[action]
        next_state = (state[0] + dx, state[1] + dy)

        # stop the move if it would go outside the grid or to the roadblock
        if not self.in_bounds(next_state) or next_state in self.roadblocks:
            return state

        return next_state

    def get_reward(self, next_state: State) -> float:
        if next_state == self.goal:
            return self.goal_reward         #goal_reward = 10
        return self.step_reward         #step_reward = -1

    def get_transition(self, state: State, action: Action) -> Tuple[State, float]:
        next_state = self.move(state, action)
        reward = self.get_reward(next_state)
        return next_state, reward

    def compute_q_value(self, state: State, action: Action, values: Dict[State, float]) -> float:
        next_state, reward = self.get_transition(state, action)
        return reward + self.gamma * values[next_state]         #bellman's equation (without the choose the max q part)

    def extract_policy(self, values: Dict[State, float]) -> Dict[State, Optional[Action]]:
        '''
        Build the policy from the value function
        '''
        policy: Dict[State, Optional[Action]] = {}

        for state in self.states:
            if self.is_terminal(state):
                policy[state] = None
                continue

            best_action = max(
                self.actions,
                key=lambda action: self.compute_q_value(state, action, values),       #bellman's equation (choose the max q part)
            )
            policy[state] = best_action

        return policy

    def value_iteration(
        self,
        threshold: float = 1e-10,
        max_iterations: int = 1000,
    ) -> Tuple[Dict[State, float], Dict[State, Optional[Action]], int]:
        values = {state: 0.0 for state in self.states}

        for iteration in range(1, max_iterations + 1):
            delta = 0.0
            updated_values = values.copy()

            for state in self.states:
                if self.is_terminal(state):
                    updated_values[state] = 0.0
                    continue

                updated_values[state] = max(
                    self.compute_q_value(state, action, values)
                    for action in self.actions
                )
                delta = max(delta, abs(updated_values[state] - values[state]))

            values = updated_values
            if delta < threshold:
                policy = self.extract_policy(values)
                return values, policy, iteration

        policy = self.extract_policy(values)
        return values, policy, max_iterations

    def policy_evaluation(
        self,
        policy: Dict[State, Optional[Action]],
        threshold: float = 1e-10,
        max_iterations: int = 1000,
    ) -> Tuple[Dict[State, float], int]:
        values = {state: 0.0 for state in self.states}

        for iteration in range(1, max_iterations + 1):
            delta = 0.0
            updated_values = values.copy()

            for state in self.states:
                if self.is_terminal(state):
                    updated_values[state] = 0.0
                    continue

                action = policy[state]
                assert action is not None
                updated_values[state] = self.compute_q_value(state, action, values)
                delta = max(delta, abs(updated_values[state] - values[state]))

            values = updated_values
            if delta < threshold:
                return values, iteration

        return values, max_iterations

    def policy_iteration(
        self,
        threshold: float = 1e-10,
        max_policy_iterations: int = 100,
        max_eval_iterations: int = 1000,
    ) -> Tuple[Dict[State, float], Dict[State, Optional[Action]], int, int]:
        policy: Dict[State, Optional[Action]] = {}
        for state in self.states:
            if self.is_terminal(state):
                policy[state] = None
            else:
                policy[state] = "U"

        total_eval_iterations = 0

        #policy evaluation
        for improvement_step in range(1, max_policy_iterations + 1):   #to alternate between policy evaluation and policy improvement
            values, eval_iterations = self.policy_evaluation(
                policy,
                threshold=threshold,
                max_iterations=max_eval_iterations,
            )
            total_eval_iterations += eval_iterations

            policy_stable = True
            improved_policy = policy.copy()

            #policy improvement
            for state in self.states:
                if self.is_terminal(state):
                    continue

                old_action = policy[state]
                best_action = max(
                    self.actions,
                    key=lambda action: self.compute_q_value(state, action, values),
                )
                improved_policy[state] = best_action

                if best_action != old_action:
                    policy_stable = False

            policy = improved_policy

            if policy_stable:                    #if the policy is stable (no change in action), then stop iterating and choose that policy
                return values, policy, improvement_step, total_eval_iterations

        return values, policy, max_policy_iterations, total_eval_iterations

    def format_values(self, values: Dict[State, float]) -> str:
        '''
        To format the output into a printable grid
        '''
        rows: List[str] = []
        for y in range(self.height - 1, -1, -1):
            cells: List[str] = []
            for x in range(self.width):
                state = (x, y)
                if state in self.roadblocks:
                    cells.append("#####".rjust(8))
                elif state == self.goal:
                    cells.append("GOAL".rjust(8))
                else:
                    cells.append(f"{values[state]:8.3f}")
            rows.append(" ".join(cells))
        return "\n".join(rows)

    def format_policy(self, policy: Dict[State, Optional[Action]]) -> str:
        '''
        Print the policy as a grid
        '''
        arrow_map = {
            "U": "↑",
            "D": "↓",
            "L": "←",
            "R": "→",
            None: "G",
        }

        rows: List[str] = []
        for y in range(self.height - 1, -1, -1):
            cells: List[str] = []
            for x in range(self.width):
                state = (x, y)
                if state in self.roadblocks:
                    cells.append("  #  ")
                elif state == self.goal:
                    cells.append("  G  ")
                else:
                    cells.append(f"  {arrow_map[policy[state]]}  ")
            rows.append("".join(cells))
        return "\n".join(rows)


