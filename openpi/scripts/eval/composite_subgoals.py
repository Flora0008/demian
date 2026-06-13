"""Per-subgoal success checkers and dynamic subgoal plans for composite tasks.

Two registries:
  * SUBGOAL_CHECKERS  — simple per-step evaluation for SR tracking (always on)
  * TASK_SUBGOAL_PLAN — ordered list of phases used when `--subgoal-driven`
                        eval mode is enabled. Each phase has an
                        instruction/reasoning and a `trigger_fn` that advances
                        the cursor when the milestone is reached.

The trigger check is deliberately more lenient than robocasa's strict success
condition so we advance to the next phase as soon as the model has "basically"
completed the current one, instead of waiting for an exact placement.
"""
from __future__ import annotations

from typing import Callable


# ---------------------------------------------------------------------------
# Geometry helpers (small set — used by both checkers and plans)
# ---------------------------------------------------------------------------

def _mug_pour_distances(env):
    """Return (xy_dist, z_dist) from the mug to the coffee machine pour site."""
    import numpy as np
    cm = env.coffee_machine
    obj_pos = np.array(env.sim.data.body_xpos[env.obj_body_id["obj"]])
    pour_site_name = f"{cm.naming_prefix}receptacle_place_site"
    site_id = env.sim.model.site_name2id(pour_site_name)
    pour_pos = env.sim.data.site_xpos[site_id]
    xy = float(np.linalg.norm(obj_pos[0:2] - pour_pos[0:2]))
    z = float(abs(obj_pos[2] - pour_pos[2]))
    return xy, z


# ---------------------------------------------------------------------------
# Per-step SR checkers (existing API, always evaluated)
# ---------------------------------------------------------------------------

def _prepare_coffee_subgoals(env) -> dict[str, bool]:
    """PrepareCoffee — pick mug from cabinet, place under dispenser, press start.

    robocasa's strict check uses a 4 cm xy threshold. We also expose a 10 cm
    "mug_near_dispenser" milestone since visually the mug can look placed but
    miss the tight strict threshold.
    """
    cm = env.coffee_machine
    strict = bool(cm.check_receptacle_placement_for_pouring(env, "obj"))
    try:
        xy, z = _mug_pour_distances(env)
        near = xy < 0.10 and z < 0.15
    except Exception:
        near = False
    return {
        "mug_near_dispenser": near,
        "mug_under_dispenser": strict,
        "machine_turned_on": bool(cm._turned_on),
    }


def _kettle_boiling_subgoals(env) -> dict[str, bool]:
    """KettleBoiling — place kettle on a stove burner, then turn the burner on."""
    from robocasa.utils import object_utils as OU
    try:
        kettle_on_stove = bool(OU.check_obj_fixture_contact(env, "obj", env.stove))
    except Exception:
        kettle_on_stove = False
    any_burner_on = False
    try:
        knobs = env.stove.get_knobs_state(env=env)
        for loc in knobs:
            if env.stove.is_burner_on(env=env, burner_loc=loc):
                any_burner_on = True
                break
    except Exception:
        pass
    return {
        "kettle_on_stove": kettle_on_stove,
        "any_burner_on": any_burner_on,
    }


def _thaw_in_sink_subgoals(env) -> dict[str, bool]:
    """ThawInSink — place food in sink, then turn on the faucet."""
    from robocasa.utils import object_utils as OU
    try:
        obj_in_sink = bool(OU.obj_inside_of(env, "obj", env.sink))
    except Exception:
        obj_in_sink = False
    try:
        faucet_on = bool(env.sink.get_handle_state(env=env).get("water_on", False))
    except Exception:
        faucet_on = False
    return {
        "obj_in_sink": obj_in_sink,
        "faucet_on": faucet_on,
    }


def _start_electric_kettle_subgoals(env) -> dict[str, bool]:
    """StartElectricKettle — close the lid, then press the on lever."""
    try:
        state = env.electric_kettle.get_state(env)
        lid = float(state.get("lid", 1.0))
        turned_on = bool(state.get("turned_on", False))
    except Exception:
        lid, turned_on = 1.0, False
    return {
        "lid_nearly_closed": lid <= 0.10,  # lenient
        "lid_closed": lid <= 0.01,         # strict (matches robocasa)
        "kettle_turned_on": turned_on,
    }


def _kettle_near_burner(env, th: float = 0.25) -> bool:
    """True iff kettle is within `th` metres of at least one burner site.

    More informative than just 'kettle touches stove' because the stove is big.
    """
    import numpy as np
    try:
        kettle = env.objects["obj"]
        kettle_pos = np.array(env.sim.data.body_xpos[env.obj_body_id[kettle.name]])[0:2]
        for loc, site in env.stove.burner_sites.items():
            if site is None:
                continue
            burner_pos = np.array(env.sim.data.get_site_xpos(site.get("name")))[0:2]
            if float(np.linalg.norm(burner_pos - kettle_pos)) < th:
                return True
    except Exception:
        pass
    return False


def _choose_measuring_cup_subgoals(env) -> dict[str, bool]:
    """ChooseMeasuringCup — open drawer, then pick correct cup to counter."""
    from robocasa.utils import object_utils as OU
    try:
        drawer_open_any = bool(env.drawer.is_open(env=env, th=0.30))
        drawer_open_full = bool(env.drawer.is_open(env=env))
    except Exception:
        drawer_open_any, drawer_open_full = False, False
    try:
        large_on_counter = bool(OU.check_obj_fixture_contact(env, "measuring_cup_large", env.counter))
        small_on_counter = bool(OU.check_obj_fixture_contact(env, "measuring_cup_small", env.counter))
    except Exception:
        large_on_counter, small_on_counter = False, False
    any_on_counter = large_on_counter or small_on_counter
    # Which cup is "correct" depends on ep's use_tablespoon flag (sometimes present).
    try:
        use_tbsp = bool(getattr(env, "use_tablespoon", False))
    except Exception:
        use_tbsp = False
    correct_cup_on_counter = large_on_counter if use_tbsp else small_on_counter
    return {
        "drawer_partly_open": drawer_open_any,
        "drawer_fully_open": drawer_open_full,
        "any_cup_on_counter": any_on_counter,
        "correct_cup_on_counter": correct_cup_on_counter,
    }


def _frying_pan_adjustment_subgoals(env) -> dict[str, bool]:
    """FryingPanAdjustment — move pan to a different burner, then turn it on."""
    pan_on_new_burner = False
    any_burner_on = False
    try:
        curr = env._check_obj_location_on_stove("obj")
        start = getattr(env, "start_loc", None)
        pan_on_new_burner = curr is not None and curr != start
    except Exception:
        pass
    try:
        knobs = env.stove.get_knobs_state(env=env)
        for loc in knobs:
            if env.stove.is_burner_on(env=env, burner_loc=loc):
                any_burner_on = True
                break
    except Exception:
        pass
    return {
        "pan_on_new_burner": pan_on_new_burner,
        "any_burner_on": any_burner_on,
    }


def _snack_sorting_subgoals(env) -> dict[str, bool]:
    """SnackSorting — bar in bowl, then close drawer."""
    from robocasa.utils import object_utils as OU
    try:
        bar_in_bowl = bool(OU.check_obj_in_receptacle(env, "bar", "bowl"))
    except Exception:
        bar_in_bowl = False
    try:
        drawer_closed = bool(env.drawer.is_closed(env=env))
    except Exception:
        drawer_closed = False
    return {
        "bar_in_bowl": bar_in_bowl,
        "drawer_closed": drawer_closed,
    }


def _wash_lettuce_subgoals(env) -> dict[str, bool]:
    """WashLettuce — turn on faucet, then hold lettuce under water for 25 steps."""
    try:
        faucet_on = bool(env.sink.get_handle_state(env=env).get("water_on", False))
    except Exception:
        faucet_on = False
    try:
        lettuce_under = bool(env.sink.check_obj_under_water(env, "lettuce"))
    except Exception:
        lettuce_under = False
    try:
        washed_time = int(getattr(env, "washed_time", 0))
    except Exception:
        washed_time = 0
    return {
        "faucet_on": faucet_on,
        "lettuce_under_water": lettuce_under,
        "washed_complete": washed_time >= 25,
    }


def _safe_in_receptacle(env, obj, recep) -> bool:
    from robocasa.utils import object_utils as OU
    try:
        return bool(OU.check_obj_in_receptacle(env, obj, recep))
    except Exception:
        return False


def _safe_inside_of(env, obj, fixture) -> bool:
    from robocasa.utils import object_utils as OU
    try:
        return bool(OU.obj_inside_of(env, obj, fixture))
    except Exception:
        return False


def _safe_fixture_contact(env, obj, fixture) -> bool:
    from robocasa.utils import object_utils as OU
    try:
        return bool(OU.check_obj_fixture_contact(env, obj, fixture))
    except Exception:
        return False


def _cupcake_cleanup_subgoals(env) -> dict[str, bool]:
    return {
        "bowl_in_sink": _safe_inside_of(env, "bowl", env.sink),
        "cupcake_on_counter": _safe_fixture_contact(env, "cupcake", env.counter),
    }


def _add_ice_cubes_subgoals(env) -> dict[str, bool]:
    n = int(getattr(env, "num_ice_cubes", 2))
    first = _safe_inside_of(env, "ice_cube0", env.blender)
    all_in = all(_safe_inside_of(env, f"ice_cube{i}", env.blender) for i in range(n))
    return {"first_ice_in_blender": first, "all_ice_in_blender": all_in}


def _bread_and_cheese_subgoals(env) -> dict[str, bool]:
    return {
        "first_item_on_board": _safe_in_receptacle(env, "obj", "container"),
        "both_items_on_board": _safe_in_receptacle(env, "obj", "container") and _safe_in_receptacle(env, "obj2", "container"),
    }


def _dessert_upgrade_subgoals(env) -> dict[str, bool]:
    return {
        "first_dessert_on_tray": _safe_in_receptacle(env, "dessert1", "receptacle"),
        "both_desserts_on_tray": _safe_in_receptacle(env, "dessert1", "receptacle") and _safe_in_receptacle(env, "dessert2", "receptacle"),
    }


def _condiment_collection_subgoals(env) -> dict[str, bool]:
    c1 = _safe_inside_of(env, "condiment1", env.cab)
    c2 = _safe_inside_of(env, "condiment2", env.cab)
    return {
        "first_condiment_in_cab": c1,
        "both_condiments_in_cab": c1 and c2,
    }


def _organize_coffee_condiments_subgoals(env) -> dict[str, bool]:
    try:
        o1 = bool(env.check_contact(env.objects["obj1"], env.coffee_machine_counter))
        o2 = bool(env.check_contact(env.objects["obj2"], env.coffee_machine_counter))
    except Exception:
        o1, o2 = False, False
    return {
        "first_item_on_counter": o1,
        "both_items_on_counter": o1 and o2,
    }


def _remove_cutting_board_items_subgoals(env) -> dict[str, bool]:
    veg_off = not _safe_in_receptacle(env, "vegetable", "cutting_board")
    meat_off = not _safe_in_receptacle(env, "meat", "cutting_board")
    return {
        "first_item_off_board": veg_off or meat_off,
        "both_items_off_board": veg_off and meat_off,
    }


def _gather_vegetables_subgoals(env) -> dict[str, bool]:
    v1 = _safe_in_receptacle(env, "veg1", "cutting_board")
    v2 = _safe_in_receptacle(env, "veg2", "cutting_board")
    return {
        "first_veg_on_board": v1,
        "both_veg_on_board": v1 and v2,
    }


def _dry_dishes_subgoals(env) -> dict[str, bool]:
    try:
        d1 = bool(env.check_contact(env.objects["obj1"], env.counter))
        d2 = bool(env.check_contact(env.objects["obj2"], env.counter))
    except Exception:
        d1, d2 = False, False
    return {
        "first_dish_on_counter": d1,
        "both_dishes_on_counter": d1 and d2,
    }


def _recycle_stacked_yogurt_subgoals(env) -> dict[str, bool]:
    one_in = _safe_in_receptacle(env, "yogurt_container2", "yogurt_container1")
    both_in = one_in and _safe_in_receptacle(env, "yogurt_container3", "yogurt_container2")
    return {
        "first_stack": one_in,
        "full_stack": both_in,
    }


SUBGOAL_CHECKERS: dict[str, Callable[[object], dict[str, bool]]] = {
    "PrepareCoffee": _prepare_coffee_subgoals,
    "KettleBoiling": _kettle_boiling_subgoals,
    "ThawInSink": _thaw_in_sink_subgoals,
    "StartElectricKettle": _start_electric_kettle_subgoals,
    "ChooseMeasuringCup": _choose_measuring_cup_subgoals,
    "FryingPanAdjustment": _frying_pan_adjustment_subgoals,
    "SnackSorting": _snack_sorting_subgoals,
    "WashLettuce": _wash_lettuce_subgoals,
    "CupcakeCleanup": _cupcake_cleanup_subgoals,
    "AddIceCubes": _add_ice_cubes_subgoals,
    "BreadAndCheese": _bread_and_cheese_subgoals,
    "DessertUpgrade": _dessert_upgrade_subgoals,
    "CondimentCollection": _condiment_collection_subgoals,
    "OrganizeCoffeeCondiments": _organize_coffee_condiments_subgoals,
    "RemoveCuttingBoardItems": _remove_cutting_board_items_subgoals,
    "GatherVegetables": _gather_vegetables_subgoals,
    "DryDishes": _dry_dishes_subgoals,
    "RecycleStackedYogurt": _recycle_stacked_yogurt_subgoals,
}


def get_subgoal_checker(task_name: str) -> Callable[[object], dict[str, bool]] | None:
    """Return a per-step SR checker for the task, or None if not registered."""
    return SUBGOAL_CHECKERS.get(task_name)


# ---------------------------------------------------------------------------
# Subgoal-driven plans — used only when --subgoal-driven is set at eval time
# ---------------------------------------------------------------------------

def _trigger_mug_near_dispenser(env) -> bool:
    try:
        xy, z = _mug_pour_distances(env)
        return xy < 0.10 and z < 0.15
    except Exception:
        return False


def _trigger_machine_on(env) -> bool:
    try:
        return bool(env.coffee_machine._turned_on)
    except Exception:
        return False


# --- KettleBoiling triggers ---

def _trigger_kettle_near_burner(env) -> bool:
    """Tightened from 'touching stove' to 'within 25 cm of any burner site'.

    Earlier the loose trigger caused the robot to drop the kettle when the
    phase swapped while the kettle was still being carried onto the stove.
    """
    return _kettle_near_burner(env, th=0.25)


def _trigger_any_burner_on(env) -> bool:
    try:
        knobs = env.stove.get_knobs_state(env=env)
        for loc in knobs:
            if env.stove.is_burner_on(env=env, burner_loc=loc):
                return True
    except Exception:
        pass
    return False


# --- ThawInSink triggers ---

def _trigger_obj_in_sink(env) -> bool:
    from robocasa.utils import object_utils as OU
    try:
        return bool(OU.obj_inside_of(env, "obj", env.sink))
    except Exception:
        return False


def _trigger_faucet_on(env) -> bool:
    try:
        return bool(env.sink.get_handle_state(env=env).get("water_on", False))
    except Exception:
        return False


# --- StartElectricKettle triggers ---

def _trigger_lid_nearly_closed(env) -> bool:
    try:
        return float(env.electric_kettle.get_state(env)["lid"]) <= 0.10
    except Exception:
        return False


def _trigger_kettle_turned_on(env) -> bool:
    try:
        return bool(env.electric_kettle.get_state(env)["turned_on"])
    except Exception:
        return False


# --- ChooseMeasuringCup triggers ---

def _trigger_drawer_open_80pct(env) -> bool:
    try:
        return bool(env.drawer.is_open(env=env, th=0.80))
    except Exception:
        return False


def _trigger_correct_cup_on_counter(env) -> bool:
    from robocasa.utils import object_utils as OU
    try:
        use_tbsp = bool(getattr(env, "use_tablespoon", False))
        target = "measuring_cup_large" if use_tbsp else "measuring_cup_small"
        return bool(OU.check_obj_fixture_contact(env, target, env.counter))
    except Exception:
        return False


# --- FryingPanAdjustment triggers ---

def _trigger_pan_on_new_burner(env) -> bool:
    try:
        curr = env._check_obj_location_on_stove("obj")
        start = getattr(env, "start_loc", None)
        return curr is not None and curr != start
    except Exception:
        return False


# _trigger_any_burner_on already defined above — reuse for phase 2


# --- SnackSorting triggers ---

def _trigger_bar_in_bowl(env) -> bool:
    from robocasa.utils import object_utils as OU
    try:
        return bool(OU.check_obj_in_receptacle(env, "bar", "bowl"))
    except Exception:
        return False


def _trigger_drawer_closed(env) -> bool:
    try:
        return bool(env.drawer.is_closed(env=env))
    except Exception:
        return False


# --- WashLettuce triggers ---

# phase-1 trigger: faucet on — reuses _trigger_faucet_on defined above
# phase-2 trigger: lettuce has been under water for at least a few steps
def _trigger_lettuce_wet(env) -> bool:
    try:
        return int(getattr(env, "washed_time", 0)) >= 3
    except Exception:
        return False


# --- Multi-object pick-place triggers (one per new task) ---

def _trigger_bowl_in_sink(env) -> bool:
    return _safe_inside_of(env, "bowl", env.sink)


def _trigger_cupcake_on_counter(env) -> bool:
    return _safe_fixture_contact(env, "cupcake", env.counter)


def _trigger_first_ice_in_blender(env) -> bool:
    return _safe_inside_of(env, "ice_cube0", env.blender)


def _trigger_all_ice_in_blender(env) -> bool:
    n = int(getattr(env, "num_ice_cubes", 2))
    return all(_safe_inside_of(env, f"ice_cube{i}", env.blender) for i in range(n))


def _trigger_first_bread_cheese(env) -> bool:
    return _safe_in_receptacle(env, "obj", "container")


def _trigger_both_bread_cheese(env) -> bool:
    return _safe_in_receptacle(env, "obj", "container") and _safe_in_receptacle(env, "obj2", "container")


def _trigger_first_dessert(env) -> bool:
    return _safe_in_receptacle(env, "dessert1", "receptacle")


def _trigger_both_desserts(env) -> bool:
    return _safe_in_receptacle(env, "dessert1", "receptacle") and _safe_in_receptacle(env, "dessert2", "receptacle")


def _trigger_first_condiment_in_cab(env) -> bool:
    return _safe_inside_of(env, "condiment1", env.cab)


def _trigger_both_condiments_in_cab(env) -> bool:
    return _safe_inside_of(env, "condiment1", env.cab) and _safe_inside_of(env, "condiment2", env.cab)


def _trigger_first_coffee_condiment(env) -> bool:
    try:
        return bool(env.check_contact(env.objects["obj1"], env.coffee_machine_counter))
    except Exception:
        return False


def _trigger_both_coffee_condiments(env) -> bool:
    try:
        return bool(env.check_contact(env.objects["obj1"], env.coffee_machine_counter)) and \
               bool(env.check_contact(env.objects["obj2"], env.coffee_machine_counter))
    except Exception:
        return False


def _trigger_first_off_board(env) -> bool:
    return (not _safe_in_receptacle(env, "vegetable", "cutting_board")) or \
           (not _safe_in_receptacle(env, "meat", "cutting_board"))


def _trigger_both_off_board(env) -> bool:
    return (not _safe_in_receptacle(env, "vegetable", "cutting_board")) and \
           (not _safe_in_receptacle(env, "meat", "cutting_board"))


def _trigger_first_veg_on_board(env) -> bool:
    return _safe_in_receptacle(env, "veg1", "cutting_board")


def _trigger_both_veg_on_board(env) -> bool:
    return _safe_in_receptacle(env, "veg1", "cutting_board") and _safe_in_receptacle(env, "veg2", "cutting_board")


def _trigger_first_dish_on_counter(env) -> bool:
    try:
        return bool(env.check_contact(env.objects["obj1"], env.counter))
    except Exception:
        return False


def _trigger_both_dishes_on_counter(env) -> bool:
    try:
        return bool(env.check_contact(env.objects["obj1"], env.counter)) and \
               bool(env.check_contact(env.objects["obj2"], env.counter))
    except Exception:
        return False


def _trigger_first_yogurt_stack(env) -> bool:
    return _safe_in_receptacle(env, "yogurt_container2", "yogurt_container1")


def _trigger_full_yogurt_stack(env) -> bool:
    return _safe_in_receptacle(env, "yogurt_container2", "yogurt_container1") and \
           _safe_in_receptacle(env, "yogurt_container3", "yogurt_container2")


# Reasoning text is keyed by caption type (matches gt_captions_composite.json keys).
TASK_SUBGOAL_PLAN: dict[str, dict] = {
    "PrepareCoffee": {
        "subgoals": [
            {
                "name": "place_mug_under_dispenser",
                "instruction": "pick the mug from the cabinet and place it under the coffee machine dispenser",
                "reasoning": {
                    "current_action": "The robot arm reaches toward the cabinet, grasps the mug, and places it under the coffee machine dispenser.",
                    "environment-simple": "A kitchen with a cabinet holding a mug and a coffee machine on the counter. The robot must bring the mug to the dispenser.",
                },
                "trigger_fn": _trigger_mug_near_dispenser,
            },
            {
                "name": "press_start_button",
                "instruction": "press the start button on the coffee machine",
                "reasoning": {
                    "current_action": "The robot arm moves toward the coffee machine's start button and presses it down.",
                    "environment-simple": "The coffee machine has a start button on its front. The robot must press it to begin brewing.",
                },
                "trigger_fn": _trigger_machine_on,
            },
        ],
    },
    "KettleBoiling": {
        "subgoals": [
            {
                "name": "place_kettle_on_burner",
                "instruction": "pick the kettle from the counter and place it on a stove burner",
                "reasoning": {
                    "current_action": "The robot arm grasps the kettle on the counter and moves it onto the stovetop, setting it down on one of the burners.",
                    "environment-simple": "A kitchen with a kettle on the counter and a stove with burners. The robot must move the kettle to a burner.",
                },
                "trigger_fn": _trigger_kettle_near_burner,
            },
            {
                "name": "turn_on_burner",
                "instruction": "turn on the stove burner",
                "reasoning": {
                    "current_action": "The robot arm reaches for the stove knob and rotates it to turn the burner on.",
                    "environment-simple": "The stove has knobs along its front. The robot must turn the one that controls the burner under the kettle.",
                },
                "trigger_fn": _trigger_any_burner_on,
            },
        ],
    },
    "ThawInSink": {
        "subgoals": [
            {
                "name": "place_obj_in_sink",
                "instruction": "pick the object from the counter and place it in the sink",
                "reasoning": {
                    "current_action": "The robot arm picks up the frozen food item from the counter and lowers it into the sink basin.",
                    "environment-simple": "A kitchen with a frozen food on the counter next to the sink. The robot must move the food into the sink.",
                },
                "trigger_fn": _trigger_obj_in_sink,
            },
            {
                "name": "turn_on_faucet",
                "instruction": "turn on the sink faucet",
                "reasoning": {
                    "current_action": "The robot arm reaches for the faucet handle and lifts it to start the water flow.",
                    "environment-simple": "The sink has a single-handle faucet. The robot must lift the handle to turn on the water.",
                },
                "trigger_fn": _trigger_faucet_on,
            },
        ],
    },
    "StartElectricKettle": {
        "subgoals": [
            {
                "name": "close_kettle_lid",
                "instruction": "close the lid of the electric kettle",
                "reasoning": {
                    "current_action": "The robot arm pushes down on the open lid of the electric kettle until it is closed.",
                    "environment-simple": "An electric kettle with its lid flipped open sits on the counter. The robot must close the lid.",
                },
                "trigger_fn": _trigger_lid_nearly_closed,
            },
            {
                "name": "press_on_lever",
                "instruction": "press down the lever to turn on the electric kettle",
                "reasoning": {
                    "current_action": "The robot arm presses down on the kettle's on lever until the kettle activates.",
                    "environment-simple": "The electric kettle has an on/off lever on its handle. The robot must push it down to start.",
                },
                "trigger_fn": _trigger_kettle_turned_on,
            },
        ],
    },
    "ChooseMeasuringCup": {
        "subgoals": [
            {
                "name": "open_drawer",
                "instruction": "open the drawer",
                "reasoning": {
                    "current_action": "The robot arm grasps the drawer handle and pulls it open to reveal the contents.",
                    "environment-simple": "A kitchen drawer that contains measuring cups. The robot must pull the handle to open it.",
                },
                "trigger_fn": _trigger_drawer_open_80pct,
            },
            {
                "name": "place_measuring_cup_on_counter",
                "instruction": "pick the measuring cup from the drawer and place it on the counter",
                "reasoning": {
                    "current_action": "The robot arm reaches into the open drawer, grasps the measuring cup, and lifts it onto the counter.",
                    "environment-simple": "The drawer is open. The robot must take a measuring cup out of the drawer and put it on the counter.",
                },
                "trigger_fn": _trigger_correct_cup_on_counter,
            },
        ],
    },
    "FryingPanAdjustment": {
        "subgoals": [
            {
                "name": "move_pan_to_other_burner",
                "instruction": "pick the pan and place it on a different stove burner",
                "reasoning": {
                    "current_action": "The robot arm grasps the frying pan's handle and moves the pan from its current burner to a neighbouring burner on the stove.",
                    "environment-simple": "A stove with multiple burners. A pan sits on one burner; the robot must move it to another burner.",
                },
                "trigger_fn": _trigger_pan_on_new_burner,
            },
            {
                "name": "turn_on_new_burner",
                "instruction": "turn on the stove burner",
                "reasoning": {
                    "current_action": "The robot arm reaches for the stove knob controlling the burner under the pan and turns it on.",
                    "environment-simple": "The pan is on a burner. The robot must rotate the corresponding stove knob to turn that burner on.",
                },
                "trigger_fn": _trigger_any_burner_on,
            },
        ],
    },
    "SnackSorting": {
        "subgoals": [
            {
                "name": "place_bar_in_bowl",
                "instruction": "pick the bar from the drawer and place it in the bowl",
                "reasoning": {
                    "current_action": "The robot arm reaches into the open drawer, picks up the bar, and drops it into the bowl on the counter.",
                    "environment-simple": "A kitchen with an open drawer holding a snack bar and a bowl on the counter. The robot must move the bar into the bowl.",
                },
                "trigger_fn": _trigger_bar_in_bowl,
            },
            {
                "name": "close_drawer",
                "instruction": "close the drawer",
                "reasoning": {
                    "current_action": "The robot arm pushes the open drawer back in until it is fully closed.",
                    "environment-simple": "The drawer is open. The robot must push it closed.",
                },
                "trigger_fn": _trigger_drawer_closed,
            },
        ],
    },
    "WashLettuce": {
        "subgoals": [
            {
                "name": "turn_on_sink_faucet",
                "instruction": "turn on the sink faucet",
                "reasoning": {
                    "current_action": "The robot arm reaches for the faucet handle and lifts it so water begins to flow.",
                    "environment-simple": "A kitchen sink with a single-handle faucet. The robot must lift the handle to start the water.",
                },
                "trigger_fn": _trigger_faucet_on,
            },
            {
                "name": "hold_lettuce_under_water",
                "instruction": "hold the lettuce under the running water to wash it",
                "reasoning": {
                    "current_action": "The robot arm lifts the colander holding the lettuce and positions it under the running faucet.",
                    "environment-simple": "The faucet is running. The robot must hold the lettuce under the stream to rinse it.",
                },
                "trigger_fn": _trigger_lettuce_wet,
            },
        ],
    },
    "CupcakeCleanup": {
        "subgoals": [
            {
                "name": "place_bowl_in_sink",
                "instruction": "pick the bowl from the counter and place it in the sink",
                "reasoning": {
                    "current_action": "The robot arm grasps the mixing bowl from the counter and lowers it into the sink basin."
                },
                "trigger_fn": _trigger_bowl_in_sink,
            },
            {
                "name": "place_cupcake_on_counter",
                "instruction": "pick the cupcake from the tray and place it on the counter",
                "reasoning": {
                    "current_action": "The robot arm picks up the cupcake from the baking tray and sets it down on the counter."
                },
                "trigger_fn": _trigger_cupcake_on_counter,
            },
        ],
    },
    "AddIceCubes": {
        "subgoals": [
            {
                "name": "place_first_ice_cube",
                "instruction": "pick an ice cube from the counter and place it in the blender",
                "reasoning": {
                    "current_action": "The robot arm picks up an ice cube from the counter and drops it into the blender."
                },
                "trigger_fn": _trigger_first_ice_in_blender,
            },
            {
                "name": "place_remaining_ice_cubes",
                "instruction": "pick the remaining ice cubes from the counter and place them in the blender",
                "reasoning": {
                    "current_action": "The robot arm picks up the remaining ice cubes one by one and drops each into the blender."
                },
                "trigger_fn": _trigger_all_ice_in_blender,
            },
        ],
    },
    "BreadAndCheese": {
        "subgoals": [
            {
                "name": "place_first_item_on_board",
                "instruction": "pick the bread from the counter and place it on the cutting board",
                "reasoning": {
                    "current_action": "The robot arm grasps the bread and lays it on the cutting board."
                },
                "trigger_fn": _trigger_first_bread_cheese,
            },
            {
                "name": "place_second_item_on_board",
                "instruction": "pick the cheese from the counter and place it on the cutting board",
                "reasoning": {
                    "current_action": "The robot arm grasps the cheese and places it on the cutting board next to the bread."
                },
                "trigger_fn": _trigger_both_bread_cheese,
            },
        ],
    },
    "DessertUpgrade": {
        "subgoals": [
            {
                "name": "place_first_dessert",
                "instruction": "pick the dessert from the plate and place it on the tray",
                "reasoning": {
                    "current_action": "The robot arm picks a dessert item off a plate and places it on the tray."
                },
                "trigger_fn": _trigger_first_dessert,
            },
            {
                "name": "place_second_dessert",
                "instruction": "pick the other dessert from the plate and place it on the tray",
                "reasoning": {
                    "current_action": "The robot arm picks the remaining dessert item and places it on the tray."
                },
                "trigger_fn": _trigger_both_desserts,
            },
        ],
    },
    "CondimentCollection": {
        "subgoals": [
            {
                "name": "place_first_condiment_in_cabinet",
                "instruction": "pick a condiment from the counter and place it in the cabinet",
                "reasoning": {
                    "current_action": "The robot arm picks up one of the condiments and puts it inside the open cabinet."
                },
                "trigger_fn": _trigger_first_condiment_in_cab,
            },
            {
                "name": "place_second_condiment_in_cabinet",
                "instruction": "pick the other condiment from the counter and place it in the cabinet",
                "reasoning": {
                    "current_action": "The robot arm picks the remaining condiment and puts it inside the cabinet."
                },
                "trigger_fn": _trigger_both_condiments_in_cab,
            },
        ],
    },
    "OrganizeCoffeeCondiments": {
        "subgoals": [
            {
                "name": "place_first_item_next_to_mug",
                "instruction": "pick the first item from the cabinet and place it next to the mug",
                "reasoning": {
                    "current_action": "The robot arm picks one item from inside the cabinet and places it on the counter next to the mug."
                },
                "trigger_fn": _trigger_first_coffee_condiment,
            },
            {
                "name": "place_second_item_next_to_mug",
                "instruction": "pick the other item from the cabinet and place it next to the mug",
                "reasoning": {
                    "current_action": "The robot arm picks the remaining item from the cabinet and places it on the counter next to the mug."
                },
                "trigger_fn": _trigger_both_coffee_condiments,
            },
        ],
    },
    "RemoveCuttingBoardItems": {
        "subgoals": [
            {
                "name": "remove_first_item",
                "instruction": "pick one item off the cutting board and place it on the counter",
                "reasoning": {
                    "current_action": "The robot arm grasps one of the items on the cutting board and lifts it away onto the counter."
                },
                "trigger_fn": _trigger_first_off_board,
            },
            {
                "name": "remove_second_item",
                "instruction": "pick the remaining item off the cutting board and place it on the counter",
                "reasoning": {
                    "current_action": "The robot arm grasps the remaining item on the cutting board and moves it onto the counter."
                },
                "trigger_fn": _trigger_both_off_board,
            },
        ],
    },
    "GatherVegetables": {
        "subgoals": [
            {
                "name": "place_first_veg_on_board",
                "instruction": "pick a vegetable from the sink and place it on the cutting board",
                "reasoning": {
                    "current_action": "The robot arm picks one vegetable out of the sink and puts it on the cutting board."
                },
                "trigger_fn": _trigger_first_veg_on_board,
            },
            {
                "name": "place_second_veg_on_board",
                "instruction": "pick the other vegetable from the sink and place it on the cutting board",
                "reasoning": {
                    "current_action": "The robot arm picks the remaining vegetable from the sink and puts it on the cutting board."
                },
                "trigger_fn": _trigger_both_veg_on_board,
            },
        ],
    },
    "DryDishes": {
        "subgoals": [
            {
                "name": "place_first_dish_on_counter",
                "instruction": "pick the cup from the sink and place it on the counter",
                "reasoning": {
                    "current_action": "The robot arm picks the cup out of the sink and sets it on the counter."
                },
                "trigger_fn": _trigger_first_dish_on_counter,
            },
            {
                "name": "place_second_dish_on_counter",
                "instruction": "pick the bowl from the sink and place it on the counter",
                "reasoning": {
                    "current_action": "The robot arm picks the bowl out of the sink and sets it on the counter."
                },
                "trigger_fn": _trigger_both_dishes_on_counter,
            },
        ],
    },
    "RecycleStackedYogurt": {
        "subgoals": [
            {
                "name": "stack_second_container",
                "instruction": "pick a smaller yogurt container and place it on top of the larger one",
                "reasoning": {
                    "current_action": "The robot arm picks up a smaller yogurt container and places it on top of the largest container."
                },
                "trigger_fn": _trigger_first_yogurt_stack,
            },
            {
                "name": "stack_third_container",
                "instruction": "pick the smallest yogurt container and place it on top of the stack",
                "reasoning": {
                    "current_action": "The robot arm picks up the smallest yogurt container and places it on top of the stack."
                },
                "trigger_fn": _trigger_full_yogurt_stack,
            },
        ],
    },
}


def get_subgoal_plan(task_name: str) -> dict | None:
    """Return the subgoal-driven plan for a task, or None if not registered."""
    return TASK_SUBGOAL_PLAN.get(task_name)


def resolve_subgoal_runtime(
    plan: dict,
    caption_key: str | None,
    original_instruction: str,
    original_reasoning: str | None,
    swap_mode: str,
) -> list[dict]:
    """Convert a static plan into a list of per-phase runtime dicts.

    Each returned dict has keys: name, instruction, reasoning, trigger_fn.

    swap_mode controls what changes per phase:
      - "prompt":    swap instruction; reasoning stays fixed (original_reasoning)
      - "reasoning": swap reasoning; instruction stays fixed (original_instruction)
      - "both":      swap both
    """
    phases = []
    for sg in plan["subgoals"]:
        if swap_mode in ("prompt", "both"):
            instr = sg["instruction"]
        else:
            instr = original_instruction
        if swap_mode in ("reasoning", "both"):
            reas_map = sg.get("reasoning", {})
            reas = reas_map.get(caption_key) if caption_key else None
            if not reas:
                # Fall back to first available reasoning if requested key not present.
                reas = next(iter(reas_map.values()), None) if reas_map else None
        else:
            reas = original_reasoning
        phases.append({
            "name": sg["name"],
            "instruction": instr,
            "reasoning": reas,
            "trigger_fn": sg["trigger_fn"],
        })
    return phases
