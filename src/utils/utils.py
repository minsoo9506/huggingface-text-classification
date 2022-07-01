from typing import List, Tuple


def read_text(fn: str) -> Tuple[List[str], List[str]]:
    """return labels, texts from data

    Parameters
    ----------
    fn : str
        file name

    Returns
    -------
    Tuple[List[str], List[str]]
        labels, texts
    """
    with open(fn, "r") as f:
        lines = f.readlines()  # 개행문자를 기준으로 한줄씩 읽어서 리스트에 넣는다

    labels, texts = [], []
    for line in lines:
        if line.strip() != "":
            label, text = line.strip().split("\t")
            labels += [label]
            texts += [text]

    return labels, texts


if __name__ == "__main__":
    labels, texts = read_text("../../data/review.sorted.uniq.refined.shuf.tsv")
    print("[sample data]")
    print(f"label = {labels[0]}")
    print(f"text = {texts[0]}")
