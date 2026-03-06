import sys

from GridWorld import GridWorld

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    environment = GridWorld(gamma=0.9)

    vi_values, vi_policy, vi_iterations = environment.value_iteration()
    pi_values, pi_policy, pi_improvements, pi_eval_iterations = environment.policy_iteration()

    print("Grid World Task 1")
    print(f"Start state: {environment.start}")
    print(f"Goal state: {environment.goal}")
    print(f"Roadblocks: {sorted(environment.roadblocks)}")
    print(f"Discount factor: {environment.gamma}")

    print("\nValue Iteration")
    print(f"Converged in {vi_iterations} iterations.")
    print(environment.format_values(vi_values))
    print("\nOptimal policy from Value Iteration")
    print(environment.format_policy(vi_policy))

    print("\nPolicy Iteration")
    print(
        f"Converged after {pi_improvements} policy improvement steps "
        f"and {pi_eval_iterations} policy evaluation sweeps."
    )
    
    print(environment.format_values(pi_values))
    print("\nOptimal policy from Policy Iteration")
    print(environment.format_policy(pi_policy))

    same_policy = all(vi_policy[state] == pi_policy[state] for state in environment.states)
    max_value_diff = max(abs(vi_values[state] - pi_values[state]) for state in environment.states)

    print("\nComparison")
    print(f"Policies identical: {same_policy}")
    print(f"Maximum absolute difference between value functions: {max_value_diff:.12f}")


if __name__ == "__main__":
    main()
