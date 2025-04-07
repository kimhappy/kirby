import os
import sys
import wandb
from dotenv import load_dotenv

def main() -> int:
    load_dotenv()
    run = wandb.init(
        entity  = os.getenv('ENTITY' ),
        project = os.getenv('PROJECT'),
        id      = 'initialize'        ,
        resume  = 'allow')
    run.log_artifact(wandb.Artifact('audio'     , type = 'dataset'))
    run.log_artifact(wandb.Artifact('state_dict', type = 'model'  ))
    run.finish()
    return 0

if __name__ == '__main__':
    sys.exit(main())
