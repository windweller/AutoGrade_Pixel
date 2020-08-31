import matplotlib
import matplotlib.pyplot as plt
import csv

def plot_train_curve():
    with open("./output/video/r3d_18_top_10_programs_n20_480/log.csv") as f:
        reader = csv.reader(f)
        train_accu = []
        val_accu = []
        for line in reader:
            print(line)
            if len(line) < 2:
                continue
            if line[1] == 'train':
                train_accu.append(float(line[3]))
            elif line[1] == 'val':
                val_accu.append(float(line[3]))

    plt.plot(train_accu, label='Train accuracy')
    plt.plot(val_accu, label='Val accuracy')

    plt.legend()
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")

    plt.show()

if __name__ == '__main__':
    plot_train_curve()
