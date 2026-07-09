from pathlib import Path
import yaml
from dso.core.orchestrator import start_pipeline

def launch_campaign(username: str, campaign_id: str, campaign_iterations: int):
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    campaign_dir = project_root / "storage" / "campaigns" / username / campaign_id
    inputs_dir = campaign_dir / "inputs"
    runs_dir = campaign_dir / "runs"

    inputs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"---- Launching {campaign_id} for {username} ----")
    print(f"Workspace established at: {campaign_dir}")

    param_file = inputs_dir / "parameters.yaml"

    if not param_file.exists():
        print(f"Warning: No parameters.yaml inside {inputs_dir}")
        return
    
    with open(param_file, "r") as f:
        campaign_params = yaml.safe_load(f)

    start_pipeline(
        username=username,
        campaign_id=campaign_id,
        campaign_config=campaign_params,
        campaign_iterations=campaign_iterations
    )

if __name__ == "__main__":
    launch_campaign(username="ab12c34", campaign_id="campaign_001", campaign_iterations=25)