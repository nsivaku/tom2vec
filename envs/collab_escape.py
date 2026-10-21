from envs import common


def _killer_info(adversary):
    # Copied from CollabEscapeMDP.Game.play.
    info = "We have information that the killer is currently located in " + adversary.current_room.name + ". "
    if adversary.target_name == "":
        info += "We don't know which room the killer will move to next, but we do know that he is unaware of our current location. "
    else:
        info += "We also have information that the killer will certainly move to the room where " + adversary.target_name + " is. "
    return info


def rollout_fn(max_turns=50):
    with common.quiet_sandbox():
        from llm_coordination_agents.CollabEscapeMDP import Game
        from llm_coordination_agents.collab_escape_agent import LLMAgent

    def rollout(rng):
        with common.quiet_sandbox():
            game = Game()
            agents = [LLMAgent(0, common.EVAL_MODEL_NAME), LLMAgent(1, common.EVAL_MODEL_NAME)]
        survivors = [game.alice, game.bob]
        pairs = []
        for _ in range(max_turns):
            state = game.state.copy()
            with common.quiet_sandbox():
                adversary_choice = game.adversary.choose_greedily(state)
            killer_info = _killer_info(game.adversary)

            views, choices = [], []
            for agent in agents:
                desc = agent._state_to_description(state, killer_info)
                views.append(agent.generator_message + [{"role": "user", "content": desc}])
                choices.append(rng.choice(agent.available_actions_list))
            pairs.extend(common.both_directions(views))

            # Generator bookkeeping and moves, as in Game.play.
            reset_rooms = game.rooms.copy()
            for survivor, action in zip(survivors, choices):
                if "fix" in action and survivor.current_room.name in reset_rooms:
                    reset_rooms.pop(survivor.current_room.name)
                    survivor.last_action_is_fixing = True
                else:
                    survivor.last_action_is_fixing = False
            for room in reset_rooms.values():
                if not room.generator_fixed:
                    room.fix_count = 0
            with common.quiet_sandbox():
                for survivor, action in zip(survivors, choices):
                    survivor.move(action, game.rooms)
                game.adversary.move(adversary_choice)
                game.update_state()
                if game.check_game_over() in ("win", "loss"):
                    break
        return pairs

    return rollout
