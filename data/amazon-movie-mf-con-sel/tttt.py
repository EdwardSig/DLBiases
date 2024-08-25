import pandas as pd

if __name__ == "__main__":
    file = 'train.csv'
    data = pd.read_csv(file)
    train_data = data[data['sel_label'] == 1]
    train_sel = data[data['sel_label'] == 0]

    train_data.to_csv(file, index=False)
    train_sel.to_csv('train_sel.csv', index=False)
