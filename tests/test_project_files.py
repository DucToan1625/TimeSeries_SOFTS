from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = ROOT / "softs_pm25_deploy"


def test_app_exists():
    assert (DEPLOY_DIR / "app.py").is_file()


def test_requirements_exists():
    assert (DEPLOY_DIR / "requirements.txt").is_file()


def test_softs_model_exists():
    assert (ROOT / "models" / "SOFTS.py").is_file()


def test_architecture_image_exists():
    image_path = ROOT / "imgs" / "dinov3_softs_architecture.png"
    assert image_path.is_file()