from model import run_albert
from model import run_mobile_reacher
import warnings

if __name__ == "__main__":
    # show_warnings = False
    # warning_flag = "default" if show_warnings else "ignore"
    # with warnings.catch_warnings():
    #     warnings.filterwarnings(warning_flag)
    #     run_albert(render=True)

    run_mobile_reacher(render=True)