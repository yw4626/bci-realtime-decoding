from dataclasses import dataclass


@dataclass
class ProjectConfig:
    sample_rate_hz: int = 1000
    window_ms: int = 300
    step_ms: int = 50
    val_fraction: float = 0.2
    # 0 = 不限制，使用训练段内全部滑窗（推荐）。>0 时按 train_window_sampling 在全时段抽样。
    max_train_windows: int = 0
    # 验证集最多使用的滑窗数；0 = 验证段全部滑窗。
    max_val_windows: int = 2000
    # sequential=仅取最前一段；uniform=随机；stratified=按时间分桶每层抽样（长录音+限额时更稳）
    train_window_sampling: str = "stratified"
    ridge_alpha: float = 1.0
    random_state: int = 42
    model_dir: str = "artifacts"
    # 与 extract_window_features 对齐：默认仅 Hz 生理频带（alpha/beta/low_gamma/high_gamma），与 MLP/Ridge 共用；
    # 旧版按 rFFT 索引的三段功率可另选开启。
    include_legacy_fft_bands: bool = False
    include_physiological_bands: bool = True

    @property
    def window_size(self) -> int:
        return int(self.sample_rate_hz * self.window_ms / 1000)

    @property
    def step_size(self) -> int:
        return int(self.sample_rate_hz * self.step_ms / 1000)

