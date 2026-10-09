"""JAX modules are imported by OpenPI training; cache scripts do not need them."""

__all__ = [
    "ActionConditionAdapter",
    "EffectPredictor",
    "Pi0EffectPolicy",
    "RobotGrounding",
]


def __getattr__(name):
    if name == "EffectPredictor":
        from effect_vla.model.effect_query import EffectPredictor

        return EffectPredictor
    if name == "RobotGrounding":
        from effect_vla.model.robot_grounding import RobotGrounding

        return RobotGrounding
    if name == "ActionConditionAdapter":
        from effect_vla.model.action_condition_adapter import ActionConditionAdapter

        return ActionConditionAdapter
    if name == "Pi0EffectPolicy":
        from effect_vla.model.pi05_effect_policy import Pi0EffectPolicy

        return Pi0EffectPolicy
    raise AttributeError(name)
