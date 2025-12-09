from yolo_train import main as train_model

def main():
    # train_model("face-detection-1", [i for i in range(1) if i not in []], base="yolo12n", augment=False)
    train_model("face-detection-1", [i for i in range(1) if i not in []], base="yolo12s", augment=False)
    # train_model("round-2-1", [i for i in range(9) if i not in []], base="yolo12s")
    # train_model("round-2-1", [i for i in range(9) if i not in [8]], base="yolo12s")
    # train_model("Military-Base-Object-Detection-16", [i for i in range(12) if i not in []], base="yolo12s")
    # train_model("Military-Base-Object-Detection-16", [i for i in range(12) if i not in []], base="yolo12m")

if __name__ == "__main__":
    main()