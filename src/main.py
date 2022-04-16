import logging
from pathlib import Path
from myLog import Log
from datetime import datetime
import time
from datetime import timedelta
from preprocess import ProcessCARD, ProcessSTRING
from viz import Viz
from gnn import GNN


class AARG():
    """
    Main class: take the input directory and run the pipeline
    """

    def __init__(self):
        self.arg_threshold = 50
        self.string_threshold = 75

        date = datetime.today().strftime("%m-%d--%H-%M-%S")
        Log(path=Path("log"), date=date)

        date = "04-13--19-08-08"
        self.out_dir = Path("results")\
            .joinpath(date)
        Path.mkdir(self.out_dir, parents=True, exist_ok=True)
        # logging set up
        start = time.time()
        self.pipeline()

        running_time = time.time() - start
        logging.info(
            f"Total running time: {str(timedelta(seconds=running_time))}\n")

    def pipeline(self):
        try:
            card_seq, card_ls, card_map = ProcessCARD(
                Path("CARD"), self.arg_threshold, self.out_dir).process()
            graph = ProcessSTRING(Path("STRING"), card_seq, card_ls, card_map,
                                  self.string_threshold, self.out_dir).process()
            Viz(graph, self.out_dir).generate()

            parameters = {"channels_1": 64,         # Number of channels in the first layer
                          "channels_2": 32,         # Number of channels in the second layer
                          "dropout": 0.5,           # Dropout rate for the features
                          "l2_reg": 5e-4,           # L2 regularization rate
                          "learning_rate": 1e-2,    # Learning rate
                          "epochs": 200,            # Number of training epochs
                          "es_patience": 10}        # Patience for early stopping}

            # model = GNN(graph, parameters, self.out_dir).train()

        except Exception as e:
            logging.info(e)


def main():
    """
    Main function
    """
    AARG()
    ###################################################################


if __name__ == "__main__":
    main()
