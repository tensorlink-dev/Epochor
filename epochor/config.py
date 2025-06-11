
class EpochorConfig:
    def __init__(self):
        self.reward_exponent = 1.0
        self.reward_temperature = 1.0
        self.reward_strategy = "linear"
        self.first_place_boost = 1.0
        self.min_weight_threshold = 0.0
        self.bootstrap_samples = 100
        self.ci_alpha = 0.05

EPOCHOR_CONFIG = EpochorConfig()
