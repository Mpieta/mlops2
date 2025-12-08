import os
import pytest

from src.settings import Settings
from dotenv import load_dotenv


@pytest.fixture()
def export_envs() -> None:
    env_file = os.path.join("config", ".env.test")
    load_dotenv(dotenv_path=env_file)


def test_settings(export_envs) -> None:
    settings = Settings(
        ENVIRONMENT=os.environ["ENVIRONMENT"],
        APP_NAME=os.environ["APP_NAME"],
        KEY=os.environ["KEY"],
    )
    assert settings.ENVIRONMENT == "test"
    assert settings.APP_NAME == "App"
    assert settings.KEY == "123abc456d"
