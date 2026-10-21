from envs import common


def rollout_fn(players=2):
    from hanabi_learning_environment import pyhanabi

    with common.quiet_sandbox():
        from llm_coordination_agents.hanabi_agent import LLMAgent

    def new_agents():
        with common.quiet_sandbox():
            agents = [LLMAgent(0, common.EVAL_MODEL_NAME), LLMAgent(1, common.EVAL_MODEL_NAME)]
        for a in agents:
            a.no_interpretation_ablation = True
        return agents

    def describe(agent, observation, agents):
        episodic = [agents[0].action_history, agents[1].action_history]
        working = [agents[0].working_memory, agents[1].working_memory]
        with common.quiet_sandbox():
            text = agent._observation_to_description(observation, episodic, working)
        return agent.base_message + [{"role": "user", "content": text}]

    def rollout(rng):
        game = pyhanabi.HanabiGame({
            "players": players, "random_start_player": False,
            "seed": rng.randrange(2**31 - 1),
        })
        state = game.new_initial_state()
        agents = new_agents()
        pairs = []
        while not state.is_terminal():
            if state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
                state.deal_random_card()
                continue
            cur = state.cur_player()
            # The benchmark's run_game drives player p with agents[p - 1].
            main_agent, partner_agent = agents[cur - 1], agents[cur]
            partner_view = describe(partner_agent, state.observation(1 - cur), agents)
            # Render the mover last so its legal-move list matches this state.
            main_view = describe(main_agent, state.observation(cur), agents)
            pairs.append(common.pair(main_view, [partner_view]))

            idx = rng.randrange(len(main_agent.transformed))
            main_agent.action_history.append(main_agent.transformed[idx].title())
            state.apply_move(state.observation(cur).legal_moves()[idx])
        return pairs

    return rollout
