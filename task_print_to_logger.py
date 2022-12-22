import logging
from clearml import Task
from config import TrainingConfig

logger = logging.getLogger('task_logger')
logging.basicConfig(level=logging.INFO)


def main():
    print('HERE')
    logger.info(f'Logger here')
    config = TrainingConfig.parse_raw("config.yaml")

    with open("Dockerfile", "r") as f:
        data = f.readlines()
    data = [s.strip() for s in data if s != "\n"]

    # connect packages
    Task.add_requirements('./requirements.txt')

    task = Task.init(
        project_name=config.clearml_project,
        task_name=config.clearml_task_name,
        output_uri=True,
        auto_connect_arg_parser=False,
        auto_connect_frameworks={'pytorch': False}
        # We disconnect pytorch auto-detection, because we added manual model save points in the code
    )

    # task.set_base_docker(
    #     docker_image=data[0],
    #     docker_setup_bash_script=data[1:]
    # )
    task.connect(config, 'Args')
    task_id = task.get_parameter("Args/task_id")

    if task_id is not None:
        dataset_id = Task.get_task(task_id).get_parameters()["Args/dataset_id"]
        dataset_path = f"clearml://{dataset_id}"
        config.data = dataset_path

    logger.info(f'{config=}')
    logger.info(f'{config.data=}')


if __name__ == "__main__":
    main()
