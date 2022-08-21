import os
import pandas as pd

train_avg_depth_path = '/storage/mskim/samsung/open/train/average_depth.csv'
train_depth_path = '/storage/mskim/samsung/open/train/'

def train_depth_split(train_avg_depth_path):
    train_depth = pd.read_csv(train_avg_depth_path).sort_values(by=['0']).reset_index(drop=True)
    train_depth['2'] = train_depth['0'].str.split('_').str[1]
    train_depth = train_depth[['0', '2', '1']].rename(columns={'2': '1', '1': '2'})

    train_depth_110 = pd.DataFrame()
    train_depth_120 = pd.DataFrame()
    train_depth_130 = pd.DataFrame()
    train_depth_140 = pd.DataFrame()

    for index in range(len(train_depth.index)):
        if train_depth.iloc[index, 1] == '110':
            train_depth_110 = train_depth_110.append(train_depth.iloc[index, :])

        elif train_depth.iloc[index, 1] == '120':
            train_depth_120 = train_depth_120.append(train_depth.iloc[index, :])

        elif train_depth.iloc[index, 1] == '130':
            train_depth_130 = train_depth_130.append(train_depth.iloc[index, :])

        elif train_depth.iloc[index, 1] == '140':
            train_depth_140 = train_depth_140.append(train_depth.iloc[index, :])

    train_depth_110.to_csv('/storage/mskim/samsung/open/train/train_depth_110.csv', index=False)
    train_depth_120.to_csv('/storage/mskim/samsung/open/train/train_depth_120.csv', index=False)
    train_depth_130.to_csv('/storage/mskim/samsung/open/train/train_depth_130.csv', index=False)
    train_depth_140.to_csv('/storage/mskim/samsung/open/train/train_depth_140.csv', index=False)

if __name__=='__main__':

    cv_train_depth_1 = pd.DataFrame()
    for i in range(1,5):
        train_depth = pd.read_csv(train_depth_path + 'train_depth_1{}0.csv'.format(int(i))).sample(frac=0.7)
        cv_train_depth_1 = cv_train_depth_1.append(train_depth)
    cv_train_depth_1 = cv_train_depth_1.sort_values(by=['0']).reset_index(drop=True)
    print(cv_train_depth_1)






