import argparse
import yaml
import os
from dotenv import load_dotenv
from settings import Settings


def export_envs(environment: str = "dev") -> None:
    env_file = os.path.join("config", f".env.{environment}")
    load_dotenv(dotenv_path=env_file)


def export_secret(path: str) -> None:
    with open(path, "r") as f:
        secrets = yaml.safe_load(f)
        os.environ["KEY"] = secrets["KEY"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load environment variables from specified.env file."
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="dev",
        help="The environment to load (dev, test, prod)",
    )
    parser.add_argument(
        "--secret",
        type=str,
        default="./secrets.yaml",
        help="Path to secrets.yaml",
    )
    args = parser.parse_args()

    export_envs(args.environment)
    export_secret(args.secret)

    settings = Settings()

    print("APP_NAME: ", settings.APP_NAME)
    print("ENVIRONMENT: ", settings.ENVIRONMENT)
    print("SECRET KEY: ", settings.KEY)
