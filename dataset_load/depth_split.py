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

def cross_validation(train_depth_path):
    os.makedirs('/storage/mskim/samsung/open/train/cv_train', exist_ok=True)
    os.makedirs('/storage/mskim/samsung/open/train/cv_test', exist_ok=True)

    for i in range(1,6):
        cv_train_depth = pd.DataFrame()
        cv_test_depth = pd.DataFrame()

        for j in range(1,5):
            train_depth = pd.read_csv(train_depth_path + 'train_depth_1{}0.csv'.format(j))
            cv = train_depth.sample(frac=0.7)
            cv_train_depth = cv_train_depth.append(cv)
            cv_test_depth = cv_test_depth.append(train_depth.drop(cv.index))

        cv_train_depth = cv_train_depth.sort_values(by=['0']).reset_index(drop=True)
        cv_test_depth = cv_test_depth.sort_values(by=['0']).reset_index(drop=True)

        print(cv_train_depth)
        print(cv_test_depth)

        os.chdir("/storage/mskim/samsung/open/train/cv_train/")
        cv_train_depth.to_csv('cv_train_{}.csv'.format(i), index=False)
        os.chdir("/storage/mskim/samsung/open/train/cv_test/")
        cv_test_depth.to_csv('cv_test_{}.csv'.format(i), index=False)

# if __name__=='__main__':







