from utils.leave_one_out import main as main_loo
from utils.viterbi import main as main_viterbi


main_loo('mmm', 0, 560, 0)  # around 12.5 minutes

main_viterbi('mmm', 0, 560, 2000)  # around 2 minutes

main_loo('sigma', 0, 560, 2000)  # not more than a day probably

main_viterbi('sigma', 0, 560, 2000)  # around 2 hours
