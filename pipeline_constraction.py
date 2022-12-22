from clearml import PipelineController


def main():
    pipe = PipelineController(
        name="belgorod print smth", project="test", version="1.0.0"
    )
    pipe.set_default_execution_queue("antifraude")

    pipe.add_step(
        name="test_logger_pipeline",
        base_task_project="test",
        base_task_name="print_smth",
        # parameter_override={
        #     "Args/task_id": "${data_creation.id}"}
    )
    pipe.start(queue="test_belgorod")


if __name__ == "__main__":
    main()
