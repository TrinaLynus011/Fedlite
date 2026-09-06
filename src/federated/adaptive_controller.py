import yaml
import numpy as np


class AdaptiveFederatedController:
    """
    Adaptive Privacy-Utility-Communication Controller.

    The controller observes completed-round telemetry and selects the
    next-round configuration for compression, local epochs, and client
    participation while respecting hard constraints from the project config.
    """

    def __init__(self, config_path="configs/adaptive_controller.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.enabled = bool(self.config.get("enabled", False))
        self.seed = int(self.config.get("seed", 42))

        self.privacy_config = dict(self.config.get("privacy", {}))
        self.comm_config = dict(self.config.get("communication", {}))
        self.utility_config = dict(self.config.get("utility", {}))
        self.resource_config = dict(self.config.get("resource", {}))
        self.candidates = dict(self.config.get("candidates", {}))
        self.objective_weights = dict(self.config.get("objective_weights", {}))

        self.history = []
        self.rng = np.random.default_rng(self.seed)
        self._validate_config()

    def _load_config(self):
        try:
            with open(self.config_path, "r") as f:
                cfg = yaml.safe_load(f)
            if cfg is None:
                return {"enabled": False}
            adaptive_cfg = cfg.get("adaptive_controller", cfg)
            if not isinstance(adaptive_cfg, dict):
                raise ValueError("Adaptive controller config must be a dictionary.")
            return adaptive_cfg
        except FileNotFoundError:
            return {"enabled": False}
        except Exception as exc:
            print(f"Warning: Failed to load adaptive controller config from {self.config_path}: {exc}")
            return {"enabled": False}

    def _validate_config(self):
        if not isinstance(self.config, dict):
            raise ValueError("adaptive_controller config must be a dictionary.")

        if not self.enabled:
            return

        ratios = self.candidates.get("compression_ratios")
        if not isinstance(ratios, list) or len(ratios) == 0:
            raise ValueError("adaptive_controller.candidates.compression_ratios must be a non-empty list.")
        for value in ratios:
            value = float(value)
            if not (0.0 < value <= 1.0):
                raise ValueError("compression_ratios must contain values in the interval (0, 1].")

        local_epochs = self.candidates.get("local_epochs")
        if not isinstance(local_epochs, list) or len(local_epochs) == 0:
            raise ValueError("adaptive_controller.candidates.local_epochs must be a non-empty list.")
        for value in local_epochs:
            if int(value) <= 0:
                raise ValueError("local_epochs must contain positive integers.")

        participation = self.candidates.get("participation_fractions")
        if not isinstance(participation, list) or len(participation) == 0:
            raise ValueError("adaptive_controller.candidates.participation_fractions must be a non-empty list.")
        for value in participation:
            value = float(value)
            if not (0.0 < value <= 1.0):
                raise ValueError("participation_fractions must contain values in the interval (0, 1].")

        budget_epsilon = self.privacy_config.get("budget_epsilon")
        if budget_epsilon is not None:
            budget_epsilon = float(budget_epsilon)
            if budget_epsilon <= 0.0:
                raise ValueError("privacy.budget_epsilon must be positive when set.")

        budget_bytes = self.comm_config.get("budget_bytes")
        if budget_bytes is not None:
            budget_bytes = float(budget_bytes)
            if budget_bytes <= 0.0:
                raise ValueError("communication.budget_bytes must be positive when set.")

        for key, value in self.objective_weights.items():
            if value is None:
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                raise ValueError(f"objective_weights.{key} must be numeric.")

    def observe(self, telemetry):
        """Record telemetry from a completed round."""
        if telemetry is None:
            telemetry = {}
        if not isinstance(telemetry, dict):
            raise TypeError("Controller telemetry must be a dictionary.")
        self.history.append(dict(telemetry))

    def _resolve_state(self, state=None):
        if state is not None:
            return dict(state)
        if not self.history:
            return {}
        return dict(self.history[-1])

    def _get_communication_bytes(self, state):
        for key in [
            "cumulative_bytes",
            "total_bytes_transmitted",
            "communication_cumulative_bytes",
            "bytes_total",
        ]:
            value = state.get(key)
            if value is not None:
                return float(value)
        return None

    def _get_utility_value(self, state, keys):
        for key in keys:
            value = state.get(key)
            if value is not None:
                return float(value)
        return None

    def _evaluate_constraints(self, candidate, state):
        """Return (is_valid, reason)."""
        privacy_budget = self.privacy_config.get("budget_epsilon")
        epsilon = self._get_utility_value(state, ["epsilon", "privacy_epsilon", "epsilon_value"])

        if self.privacy_config.get("enabled", False) and privacy_budget is not None and epsilon is not None:
            if epsilon >= float(privacy_budget):
                return False, "privacy_budget_exhausted"
            remaining = float(privacy_budget) - float(epsilon)
            if self.privacy_config.get("stop_on_budget_exhaustion", False) and remaining <= 0.05 * float(privacy_budget):
                return True, "privacy_near_budget"

        budget_bytes = self.comm_config.get("budget_bytes")
        total_bytes = self._get_communication_bytes(state)
        if budget_bytes is not None and total_bytes is not None and total_bytes >= float(budget_bytes):
            return False, "communication_budget_exhausted"

        minimum_auroc = self.utility_config.get("minimum_auroc")
        auroc = self._get_utility_value(state, ["val_auroc", "validation_auroc", "auroc"])
        if minimum_auroc is not None and auroc is not None and auroc < float(minimum_auroc):
            return False, "utility_below_threshold"

        minimum_auprc = self.utility_config.get("minimum_auprc")
        auprc = self._get_utility_value(state, ["val_auprc", "validation_auprc", "auprc"])
        if minimum_auprc is not None and auprc is not None and auprc < float(minimum_auprc):
            return False, "utility_below_threshold"

        max_round_time = self.resource_config.get("maximum_round_time_sec")
        training_time = self._get_utility_value(state, ["training_time_sec", "round_training_time_sec", "time_sec"])
        if max_round_time is not None and training_time is not None and training_time > float(max_round_time):
            return False, "resource_limit_exceeded"

        return True, "valid"

    def _score_candidate(self, candidate, state):
        """Deterministic heuristic score. Higher score is better."""
        comp_ratio = float(candidate["compression_ratio"])
        local_epochs = int(candidate["local_epochs"])
        participation = float(candidate["participation_fraction"])

        w_util = float(self.objective_weights.get("utility", 1.0))
        w_comm = float(self.objective_weights.get("communication", 1.0))
        w_res = float(self.objective_weights.get("resource", 1.0))
        w_priv = float(self.objective_weights.get("privacy", 1.0))

        max_epochs = max(1, max(int(v) for v in self.candidates.get("local_epochs", [1])))
        best_utility = self._get_utility_value(state, ["val_auroc", "validation_auroc", "auroc"])
        if best_utility is None:
            best_utility = 0.5
        utility_score = max(0.0, min(1.0, float(best_utility))) * (local_epochs / max_epochs) * participation * (comp_ratio ** 0.5)

        communication_cost = comp_ratio * participation
        resource_cost = (local_epochs / max_epochs) * participation

        epsilon = self._get_utility_value(state, ["epsilon", "privacy_epsilon", "epsilon_value"])
        privacy_budget = self.privacy_config.get("budget_epsilon")
        privacy_pressure = 0.0
        if privacy_budget is not None and epsilon is not None and float(privacy_budget) > 0:
            remaining = max(0.0, float(privacy_budget) - float(epsilon))
            privacy_pressure = 1.0 - (remaining / float(privacy_budget))
        privacy_cost = (local_epochs / max_epochs) * participation * (0.5 + privacy_pressure)

        score = (
            w_util * utility_score
            - w_comm * communication_cost
            - w_res * resource_cost
            - w_priv * privacy_cost
        )
        return float(score)

    def _candidate_space(self):
        ratios = self.candidates.get("compression_ratios", [1.0])
        epochs = self.candidates.get("local_epochs", [1])
        fracs = self.candidates.get("participation_fractions", [1.0])
        for cr in ratios:
            for le in epochs:
                for pf in fracs:
                    yield {
                        "compression_ratio": float(cr),
                        "local_epochs": int(le),
                        "participation_fraction": float(pf),
                    }

    def select_next_configuration(self, state=None):
        """Generate candidates, filter invalid ones, score, and select the best valid option."""
        state = self._resolve_state(state)

        if self.privacy_config.get("enabled", False) and self.privacy_config.get("stop_on_budget_exhaustion", False):
            privacy_budget = self.privacy_config.get("budget_epsilon")
            epsilon = self._get_utility_value(state, ["epsilon", "privacy_epsilon", "epsilon_value"])
            if privacy_budget is not None and epsilon is not None and epsilon >= float(privacy_budget):
                return None, {"action": "stop", "reason": "privacy_budget_exhausted", "constraint_status": "privacy_budget_exhausted"}

        valid_candidates = []
        invalid_candidates = []
        for candidate in self._candidate_space():
            is_valid, reason = self._evaluate_constraints(candidate, state)
            score = self._score_candidate(candidate, state) if is_valid else float("-inf")
            entry = {
                "candidate": candidate,
                "score": score,
                "constraint_status": reason,
            }
            if is_valid:
                valid_candidates.append(entry)
            else:
                invalid_candidates.append(entry)

        if not valid_candidates:
            invalid_reasons = [entry["constraint_status"] for entry in invalid_candidates]
            if not invalid_reasons:
                invalid_reasons = ["all_candidates_invalid"]
            return None, {
                "action": "stop",
                "reason": "all_candidates_violate_constraints",
                "constraint_status": invalid_reasons[0],
                "candidate_scores": [
                    {"candidate": entry["candidate"], "score": entry["score"], "constraint_status": entry["constraint_status"]}
                    for entry in invalid_candidates
                ],
            }

        selected = max(
            valid_candidates,
            key=lambda item: (item["score"], -float(item["candidate"]["compression_ratio"]), -int(item["candidate"]["local_epochs"]), -float(item["candidate"]["participation_fraction"]))
        )
        selected_candidate = dict(selected["candidate"])
        metadata = {
            "action": "continue",
            "reason": "highest_valid_score",
            "selected_score": float(selected["score"]),
            "constraint_status": "valid",
            "controller_decision_reason": "highest_valid_score",
            "candidate_scores": [
                {"candidate": entry["candidate"], "score": entry["score"], "constraint_status": entry["constraint_status"]}
                for entry in valid_candidates
            ],
        }
        return selected_candidate, metadata

