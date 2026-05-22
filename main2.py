from experiments.train_QL import run_q_learning


ALGORITHMS = {
    "q_learning": run_q_learning
}


def main():

    algorithm = "q_learning"

    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    ALGORITHMS[algorithm]()


if __name__ == "__main__":
    main()