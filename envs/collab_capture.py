from envs import common

MAX_TURNS = 30


def rollout_fn():
    with common.quiet_sandbox():
        from llm_coordination_agents.collab_capture_action_manager import Agent, Environment, Thief

    def rollout(rng):
        environment = Environment()
        environment.doors_open[1] = False
        environment.doors_open[3] = True
        with common.quiet_sandbox():
            alice = Agent(1, 0, "Alice", environment, common.EVAL_MODEL_NAME)
            bob = Agent(6, 1, "Bob", environment, common.EVAL_MODEL_NAME)
        thief = Thief(2, "Thief", environment)
        players = [alice, bob]
        pairs = []
        for _ in range(MAX_TURNS):
            state = environment.get_state_for_llm(alice, bob, thief)
            views = []
            for p in players:
                desc = p.llm_agent._state_to_description(state)
                views.append(p.llm_agent.message + [{"role": "user", "content": desc}])
            pairs.extend(common.both_directions(views))

            for p in players:
                action = rng.choice(state["available_actions"][p.name])
                if isinstance(action, int):
                    p.plan_move(action)
                elif action.startswith("Press"):
                    p.plan_press_button()
            thief.plan_move_away_from_agents(alice.next_room, bob.next_room)
            with common.quiet_sandbox():
                for p in players + [thief]:
                    p.execute_move()

            caught = any(
                thief.current_room == p.current_room
                or (p.previous_room == thief.current_room and thief.previous_room == p.current_room)
                for p in players
            )
            if caught:
                break
            for p in players + [thief]:
                p.previous_room = p.current_room
        return pairs

    return rollout
