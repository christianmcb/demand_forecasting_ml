import subprocess

def test_training_script_runs():

    result = subprocess.run(
        ["python", "-m", "scripts.train"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0