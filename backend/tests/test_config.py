from __future__ import annotations

import textwrap


def test_default_config_loads():
    from robotics_playground.config import PlaygroundConfig

    config = PlaygroundConfig()
    assert config.server.host == "0.0.0.0"
    assert config.server.port == 8000
    assert config.bridge.type == "mock"
    assert config.rerun.grpc_port == 9876


def test_load_config_from_yaml(tmp_path):
    from robotics_playground.config import load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        textwrap.dedent("""\
            server:
              port: 9000
              log_level: debug
            bridge:
              type: ros2
            ros2:
              domain_id: 5
              cameras:
                wrist: /cam/wrist
        """)
    )
    config = load_config(str(config_file))
    assert config.server.port == 9000
    assert config.server.log_level == "debug"
    assert config.bridge.type == "ros2"
    assert config.ros2.domain_id == 5
    assert config.ros2.cameras == {"wrist": "/cam/wrist"}


def test_load_config_missing_file_raises():
    import pytest

    from robotics_playground.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")


def test_env_var_sets_config_path(tmp_path, monkeypatch):
    from robotics_playground.config import load_config

    config_file = tmp_path / "env.yaml"
    config_file.write_text("server:\n  port: 7777\n")
    monkeypatch.setenv("PLAYGROUND_CONFIG", str(config_file))
    config = load_config()
    assert config.server.port == 7777


def test_ros2_config_defaults():
    from robotics_playground.config import PlaygroundConfig

    config = PlaygroundConfig()
    assert config.ros2.domain_id == 0
    assert config.ros2.discovery_server is None
    assert config.ros2.cameras == {}
    assert config.ros2.joint_state_topic == "/joint_states"
    assert config.ros2.joint_command_topic == "/joint_commands"


def test_policy_config_defaults():
    from robotics_playground.config import PlaygroundConfig

    config = PlaygroundConfig()
    assert config.policy.type == "mock"
    assert config.policy.default_model == ""
    assert config.policy.models == {}
    assert config.policy.embodiment.joint_names == []
    assert config.policy.embodiment.gripper_limits == [0.0, 0.04]


def test_policy_config_from_dict():
    from robotics_playground.config import PlaygroundConfig

    data = {
        "policy": {
            "type": "openpi",
            "default_model": "test-model",
            "models": {
                "test-model": {
                    "name": "Test Model",
                    "endpoint": "ws://localhost:8080/v1/realtime/robot/openpi",
                    "action_horizon": 4,
                }
            },
            "embodiment": {
                "joint_names": ["j1", "j2"],
                "training_order": ["j2", "j1"],
                "joint_limits": {"j1": [-1.0, 1.0], "j2": [-2.0, 2.0]},
                "gripper_joint": "grip",
                "camera_mapping": {"wrist": "observation/wrist_image_left"},
            },
        }
    }
    config = PlaygroundConfig(**data)
    assert config.policy.type == "openpi"
    assert config.policy.default_model == "test-model"
    assert "test-model" in config.policy.models
    assert (
        config.policy.models["test-model"].endpoint
        == "ws://localhost:8080/v1/realtime/robot/openpi"
    )
    assert config.policy.embodiment.joint_names == ["j1", "j2"]
    assert config.policy.embodiment.training_order == ["j2", "j1"]
    assert config.policy.embodiment.camera_mapping == {"wrist": "observation/wrist_image_left"}


def test_ros2_config_physics_decimation():
    from robotics_playground.config import PlaygroundConfig

    config = PlaygroundConfig()
    assert config.ros2.physics_decimation == 10

    config2 = PlaygroundConfig(ros2={"physics_decimation": 5})
    assert config2.ros2.physics_decimation == 5


# Tests for new ModelConfig and restructured PolicyConfig


def test_model_config_defaults():
    from robotics_playground.config import ModelConfig

    mc = ModelConfig(endpoint="ws://localhost:8080")
    assert mc.name == ""
    assert mc.endpoint == "ws://localhost:8080"
    assert mc.action_horizon == 4
    assert mc.camera_mapping is None


def test_policy_config_with_models():
    from robotics_playground.config import ModelConfig, PolicyConfig

    pc = PolicyConfig(
        type="openpi",
        default_model="m1",
        models={
            "m1": ModelConfig(name="Model 1", endpoint="ws://m1:8080"),
            "m2": ModelConfig(
                name="Model 2",
                endpoint="ws://m2:8080",
                action_horizon=8,
                camera_mapping={"cam": "observation/cam"},
            ),
        },
    )
    assert pc.default_model == "m1"
    assert len(pc.models) == 2
    assert pc.models["m2"].action_horizon == 8
    assert pc.models["m2"].camera_mapping == {"cam": "observation/cam"}


def test_embodiment_config_no_image_size():
    from robotics_playground.config import EmbodimentConfig

    ec = EmbodimentConfig()
    assert not hasattr(ec, "image_size")


def test_policy_config_no_legacy_fields():
    from robotics_playground.config import PolicyConfig

    pc = PolicyConfig()
    assert not hasattr(pc, "endpoint")
    assert not hasattr(pc, "model_name")
    assert not hasattr(pc, "action_horizon")


def test_full_config_from_dict():
    from robotics_playground.config import EmbodimentConfig, PlaygroundConfig

    data = {
        "policy": {
            "type": "openpi",
            "default_model": "dreamzero-v1",
            "models": {
                "dreamzero-v1": {
                    "name": "DreamZero",
                    "endpoint": "ws://dreamzero:8080/v1/realtime/robot/openpi",
                    "action_horizon": 4,
                },
                "pi05-v1": {
                    "name": "pi0.5",
                    "endpoint": "ws://pi05:8080/",
                    "action_horizon": 8,
                },
            },
            "embodiment": {
                "joint_names": ["j1"],
                "training_order": ["j1"],
                "joint_limits": {"j1": [-1, 1]},
                "gripper_joint": "g",
                "camera_mapping": {"wrist": "observation/wrist_image_left"},
            },
        }
    }
    cfg = PlaygroundConfig(**data)
    assert cfg.policy.models["pi05-v1"].action_horizon == 8
    assert "image_size" not in EmbodimentConfig.model_fields


def test_default_model_must_be_in_models():
    import pytest

    from robotics_playground.config import ModelConfig, PolicyConfig

    with pytest.raises(ValueError, match=r"default_model.*not found"):
        PolicyConfig(
            type="openpi",
            default_model="nonexistent",
            models={"real-v1": ModelConfig(endpoint="ws://x")},
        )


def test_default_model_empty_is_valid():
    from robotics_playground.config import ModelConfig, PolicyConfig

    cfg = PolicyConfig(
        type="openpi",
        default_model="",
        models={"real-v1": ModelConfig(endpoint="ws://x")},
    )
    assert cfg.default_model == ""
