from args import TrainingArgs
from simple_parsing import parse


def main(args: TrainingArgs):
    print("Training code")
    
if __name__ == "__main__":
    parsed_arg_groups = parse(TrainingArgs, add_config_path_arg=True)
    # current_process_rank = get_rank()
    # with graceful_exceptions(extra_message=f"Rank: {current_process_rank}"):
    main(parsed_arg_groups)
