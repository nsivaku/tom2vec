# `load_setup` copied from LLM-Deliberation/utils.py (MIT, see LICENSE) without the
# module's model-loading imports.
import os


def load_setup(game_dir, agents_num):
    '''
    load config files of the experiments (<game_dir>/config.txt)
    The config file is organized as: one line per agent. 
    Each line should be comma-separated items of: 
        1) The name of the agent (as written in the game description file), 
        2) The prefix file name of the agent (without ".txt" and without incentive)
        3) The role of the player, at the moment this can be:
            - player: no specific assigned role 
            - p2: this is p2 
            - p1: this is p1 
            - target: this is the target in the adversarial game 
        4) incentive: this is the assigned incentive (at the moment this can be cooperative, greedy, untargeted_adv, targeted_adv)
        5) model: this is the assigned model for this player 
        
    Returns:
        agents: dict of agent_name to: file, role, incentive, model 
        intial deal: deal to kick off, add as input in initial_deal_file 
        role_to_agents: dict of roles (veto) to agent names 
    '''
    with open(os.path.join(game_dir,'config.txt'), 'r') as f:
        agents_config_file = f.readlines()
        

    agents = {}
    role_to_agents = {}
    assert len(agents_config_file) == agents_num
    
    for line in agents_config_file:   
        agent_game_name, file_name, role, incentive, model = line.split(',') 
        model = model.strip()
        agents[agent_game_name]={'file_name': file_name, 'role':role, 'incentive': incentive, 'model': model}
        if not role in role_to_agents: role_to_agents[role] = []
        role_to_agents[role].append(agent_game_name)
        

    with open(os.path.join(game_dir,'initial_deal.txt'), 'r') as file: 
        initial_deal = file.readline().strip()
    
    for role in role_to_agents:
        if len(role_to_agents[role]) == 1: 
            role_to_agents[role] = role_to_agents[role][0]
        
    return agents,initial_deal,role_to_agents

