import warnings

from overcooked_ai_py.mdp.actions import Action, Direction

SYSTEM_PROMPT = """You are an agent playing the cooperative cooking game Overcooked.
You control one player in a kitchen. Your goal is to cook and deliver soups as fast as possible.

Game rules:
- You earn reward only by DELIVERING soups whose ingredient set exactly matches one of the
  currently active orders. Each delivery yields a sparse reward (default 20, can vary per layout).
- You will only be able to interact with dispensers, pots, counters, or serving areas once you
  are facing them. Movement actions turn you; INTERACT acts on the cell directly in front.
- You can only hold ONE item at a time. To pick something else up, first place what you're holding
  on a counter (X), or use it (drop ingredient into a pot, deliver soup at S, etc.).

How to cook and serve a soup (the full loop):
  1. PICK UP an ingredient: face an onion dispenser (O) or tomato dispenser (T) with empty hands
     and INTERACT. You now hold one onion / one tomato.
  2. PLACE INGREDIENT IN POT: face a pot (P) and INTERACT. The ingredient drops into the pot.
     A pot holds at most 3 ingredients ('P1' / 'P2' / 'P3' shown on the map).
  3. START COOKING: once the pot has the ingredients you want (1, 2, or 3), face the pot and
     INTERACT with empty hands to begin cooking. The pot shows 'P1*' / 'P2*' / 'P3*' while cooking
     and counts down (default 20 ticks; the prompt tells you ticks remaining).
  4. GRAB A DISH: face a dish dispenser (D) with empty hands and INTERACT to hold a dish.
  5. PLATE THE SOUP: face the cooking pot once it shows 'P!' (ready) while holding a dish, then
     INTERACT. You now hold a soup with that pot's ingredients.
  6. DELIVER: face a serving area (S) while holding the soup and INTERACT. The order is fulfilled
     and reward is paid.

Recipes / orders:
- Only onions and tomatoes can be used as ingredients. A recipe is the multiset of ingredients in
  the pot when you start cooking it. Order matters NOT, but the set must match an active order
  exactly. Common recipes are 3 onions, 3 tomatoes, or mixes (e.g. 2 onion + 1 tomato).
- The "Active orders" list in the game state shows every recipe currently worth delivering. Build
  the pot to match one of them before starting to cook.
- Adding the wrong ingredient to a pot cannot be undone; once cooking starts, the recipe is locked.

Coordination:
- Two agents share the kitchen. Avoid blocking your partner or duplicating their work (e.g. both
  carrying onions to the same pot that's already full).
- Counters (X) can be used as a handoff: drop an item on a counter for your partner to pick up.

Movement and facing:
- A movement action (NORTH/SOUTH/EAST/WEST) ALWAYS updates your facing to that direction.
  After the action, your <dir> arrow on the map will point that way.
- If the tile in that direction is walkable (' . '), you also move one cell there.
- If that tile is blocked (counter, dispenser, pot, serving area, or another player),
  you DO NOT move -- you only turn to face it. Use this to line up an INTERACT.
- To INTERACT with an object, you must first be facing it (the cell directly in front
  of your arrow). If you are not yet facing it, issue the direction action first to turn,
  then INTERACT on the next turn.
- STAY leaves both your position and your facing unchanged.

You will receive a 2D ASCII map of the kitchen each turn. Each cell is 3 characters wide.
Column indices (x) are printed above and below the grid; row indices (y) flank
each row on the left and right. Read any position as (x, y).
Terrain cells (a single character centered with spaces):
  ' . ' = walkable floor          ' X ' = counter (can place items)
  ' O ' = onion dispenser         ' T ' = tomato dispenser
  ' D ' = dish dispenser          ' P ' = empty pot
  ' S ' = serving area
Items sitting on counters / floor:
  ' o ' = onion   ' t ' = tomato   ' d ' = dish   ' s ' = soup
Pots in use:
  'P1 ' / 'P2 ' / 'P3 ' = pot with 1 / 2 / 3 ingredients (not yet cooking)
  'P1*' / 'P2*' / 'P3*' = pot currently cooking with that many ingredients
  'P! ' = pot with a ready (cooked) soup
Players occupy a 3-char cell '<idx><dir><held>':
  <idx>  = agent index (0 or 1) -- you will be told which one is you
  <dir>  = '^' (NORTH), 'v' (SOUTH), '>' (EAST), '<' (WEST)
  <held> = 'o' onion, 't' tomato, 'd' dish, 's' soup, '.' nothing
Example: '1^o' = agent 1 standing on a floor tile, facing NORTH, holding an onion.

Available actions (respond with EXACTLY one):
  NORTH  - face up; move up if unblocked       SOUTH  - face down; move down if unblocked
  EAST   - face right; move right if unblocked WEST   - face left; move left if unblocked
  STAY   - keep position and facing            INTERACT - interact with the cell you are facing

Choose the single best action given the current game state. Output only the action name, nothing else."""


def _direction_name(direction):
    return Direction.DIRECTION_TO_NAME.get(direction, str(direction))


def _action_name(action):
    if action == Action.STAY:
        return "STAY"
    if action == Action.INTERACT:
        return "INTERACT"
    return Direction.DIRECTION_TO_NAME.get(action, str(action))


def _describe_held_object(obj):
    if obj is None:
        return "nothing"
    if obj.name == "soup":
        ingredients = ", ".join(sorted(obj.ingredients)) if obj.ingredients else "no ingredients"
        if obj.is_ready:
            return f"a ready soup ({ingredients})"
        elif obj.is_cooking:
            return f"a cooking soup ({ingredients}, {obj.cook_time_remaining} ticks remaining)"
        else:
            return f"a soup pot ({ingredients}, not yet cooking)"
    return f"a {obj.name}"


_HELD_CHAR = {"onion": "o", "tomato": "t", "dish": "d", "soup": "s"}


_DIR_ARROW = {
    Direction.NORTH: "^",
    Direction.SOUTH: "v",
    Direction.EAST: ">",
    Direction.WEST: "<",
}


def _held_char(obj):
    if obj is None:
        return "."
    return _HELD_CHAR.get(obj.name, "?")


def _render_pot_cell(soup):
    """3-char pot encoding: '<onion_count><tomato_count><status>'.

    Status: '.' = ingredients present but not yet cooking,
            '*' = currently cooking, '!' = ready to be plated.
    A pot with NO ingredients renders as its terrain (' P ') since no
    SoupState exists yet. Pots hold at most 3 ingredients total.
    """
    n_onion = sum(1 for ing in soup.ingredients if ing == "onion")
    n_tomato = sum(1 for ing in soup.ingredients if ing == "tomato")
    if soup.is_ready:
        status = "!"
    elif soup.is_cooking:
        status = "*"
    else:
        status = "."
    return f"{n_onion}{n_tomato}{status}"


def _render_grid(state, mdp):
    """Render the kitchen as an ASCII map. Each cell is 3 chars wide. Rows and
    columns are labeled with their (y, x) indices for spatial reference."""
    if mdp is None or not getattr(mdp, "terrain_mtx", None):
        return None

    objects_by_pos = {obj.position: obj for obj in state.objects.values()}
    players_by_pos = {p.position: (i, p) for i, p in enumerate(state.players)}

    height = len(mdp.terrain_mtx)
    width = len(mdp.terrain_mtx[0]) if height else 0

    # Row-label column is 2 chars wide + 2-space gutter so its content aligns
    # with the start of cell 0 in the data rows below.
    row_prefix_width = 4
    col_header = " " * row_prefix_width + " ".join(f"{x:^3}" for x in range(width))
    separator = " " * row_prefix_width + "-" * (4 * width - 1) if width else ""

    rows = [col_header, separator]
    for y, terrain_row in enumerate(mdp.terrain_mtx):
        cells = []
        for x, terrain in enumerate(terrain_row):
            pos = (x, y)
            if pos in players_by_pos:
                i, p = players_by_pos[pos]
                cells.append(f"{i}{_DIR_ARROW.get(p.orientation, '?')}{_held_char(p.held_object)}")
            elif pos in objects_by_pos:
                obj = objects_by_pos[pos]
                if obj.name == "soup" and terrain == "P":
                    cells.append(_render_pot_cell(obj))
                else:
                    cells.append(f" {_held_char(obj)} ")
            elif terrain == " ":
                cells.append(" . ")
            else:
                cells.append(f" {terrain} ")
        rows.append(f"{y:>2}  " + " ".join(cells) + f"  {y:<2}")
    rows.append(separator)
    rows.append(col_header)
    return "\n".join(rows)


def _describe_recipe(ingredients):
    """Turn a recipe's ingredient tuple into 'N x ingredient' phrases (e.g. '3 onion')."""
    if not ingredients:
        return "(empty recipe)"
    counts = {}
    for ing in ingredients:
        counts[ing] = counts.get(ing, 0) + 1
    return " + ".join(f"{n} {name}" for name, n in sorted(counts.items()))


def _describe_soup_state(soup):
    ingredients = ", ".join(sorted(soup.ingredients)) if soup.ingredients else "empty"
    if soup.is_ready:
        return f"pot with ready soup ({ingredients})"
    elif soup.is_cooking:
        return f"pot cooking ({ingredients}, {soup.cook_time_remaining} ticks left)"
    elif soup.is_idle:
        return f"pot with ingredients ({ingredients}, not cooking yet)"
    return f"soup object ({ingredients})"


def state_to_text(state, agent_index, mdp=None):
    """Convert an OvercookedState to a natural language description with ASCII map."""
    lines = []
    lines.append(f"=== Game State (timestep {state.timestep}) ===")

    map_str = _render_grid(state, mdp)
    if map_str is not None:
        lines.append(f"Kitchen map (you are agent {agent_index}; coordinates are (x, y) with x=col, y=row):")
        lines.append(map_str)

    me = state.players[agent_index]
    partner_idx = 1 - agent_index
    partner = state.players[partner_idx]

    me_dir = _direction_name(me.orientation)
    me_held = _describe_held_object(me.held_object)
    lines.append(f"You (agent {agent_index}): position {me.position}, facing {me_dir}, holding {me_held}.")

    partner_dir = _direction_name(partner.orientation)
    partner_held = _describe_held_object(partner.held_object)
    lines.append(f"Partner (agent {partner_idx}): position {partner.position}, facing {partner_dir}, holding {partner_held}.")

    # Pot/soup details (cook-time remaining etc.) that the map can't fully convey
    soups = [o for o in state.objects.values() if o.name == "soup"]
    if soups:
        lines.append("Pot details:")
        for soup in soups:
            lines.append(f"  - {_describe_soup_state(soup)} at position {soup.position}")

    if state.all_orders:
        order_strs = [_describe_recipe(order.ingredients) for order in state.all_orders]
        lines.append(f"Active orders (any of these recipes earns reward on delivery): {'; '.join(order_strs)}")
    else:
        lines.append("No active orders.")
    if getattr(state, "bonus_orders", None):
        bonus_strs = [_describe_recipe(order.ingredients) for order in state.bonus_orders]
        lines.append(f"Bonus orders (extra reward): {'; '.join(bonus_strs)}")

    lines.append("What single action should you take?")
    return "\n".join(lines)


_ACTION_ALIASES = {
    "north": Direction.NORTH,
    "up": Direction.NORTH,
    "south": Direction.SOUTH,
    "down": Direction.SOUTH,
    "east": Direction.EAST,
    "right": Direction.EAST,
    "west": Direction.WEST,
    "left": Direction.WEST,
    "stay": Action.STAY,
    "wait": Action.STAY,
    "interact": Action.INTERACT,
    "pick": Action.INTERACT,
    "place": Action.INTERACT,
    "use": Action.INTERACT,
    "grab": Action.INTERACT,
    "drop": Action.INTERACT,
    "cook": Action.INTERACT,
    "serve": Action.INTERACT,
    "deliver": Action.INTERACT,
}


def parse_action(text):
    """Parse model output into a valid Action. Falls back to Action.STAY."""
    text = text.strip().lower()
    # Qwen3 thinking mode emits its reasoning before </think>; take only what
    # follows the LAST </think>. Handles outputs with or without a matching
    # opening <think> (the chat template often supplies it implicitly).
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    # Try exact or first-word match
    first_word = text.split()[0] if text.split() else ""
    for token in [first_word, text]:
        if token in _ACTION_ALIASES:
            return _ACTION_ALIASES[token]
    # Substring scan in order of specificity
    for keyword, action in _ACTION_ALIASES.items():
        if keyword in text:
            return action
    warnings.warn(f"could not parse action from: {repr(text)!r}. Defaulting to STAY.")
    return Action.STAY



def chat_messages(state, agent_index, mdp):
    """System + user messages for one agent's view of `state`."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state_to_text(state, agent_index, mdp=mdp)},
    ]


LAYOUTS = ["cramped_room", "asymmetric_advantages", "coordination_ring"]


def rollout_fn(layout, horizon=400):
    """Random-agent self-play on one layout, yielding a pair per (state,
    main agent)."""
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

    from envs.common import both_directions

    mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon)
    actions = list(Action.ALL_ACTIONS)

    def rollout(rng):
        env.reset()
        state, done, pairs = env.state, False, []
        while not done:
            views = [chat_messages(state, i, mdp) for i in range(mdp.num_players)]
            pairs.extend(both_directions(views))
            joint = [rng.choice(actions) for _ in range(mdp.num_players)]
            state, _, done, _ = env.step(joint)
        return pairs

    return rollout
