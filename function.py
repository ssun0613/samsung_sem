import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import wandb

def normal_or_error(prediction, y_spec_path):
    y_spec = pd.read_csv(y_spec_path)
    df_list=[]
    for i in range(len(y_spec.index)):
        new=[]
        for j in range(len(prediction.index)):
            if (y_spec.iloc[i, 1] <= prediction.iloc[j, i+1]) and (y_spec.iloc[i, 2] >= prediction.iloc[j, i+1]):
                prediction.iloc[j, i+1] = "O"
            else:
                prediction.iloc[j, i+1] = "X"
                new_1 = prediction.iloc[j, 0]
                # new_1 = [prediction.iloc[j, 0], prediction.columns[i+1]]
                new.append(new_1)
        y = pd.DataFrame(new, columns=['{}'.format(prediction.columns[i+1])])
        df_list.append(y)

    return prediction, df_list

def error(df_list, prediction):
    df_new = np.array(df_list[0]) # 불량에 대한 정보가 들어있는 df_list를 호출하여 np.array로 만들어줌
    for i in range(len(df_list) - 1):
        df_new = np.concatenate([df_new, np.array(df_list[i + 1])])

    df_new_1 = pd.DataFrame(df_new)
    df_new_1 = pd.DataFrame(np.array(df_new_1[~df_new_1.duplicated()]), columns=['ID'])

    for col_id in range(len(prediction.columns) - 1):
        df_new_1[prediction.columns[col_id + 1]] = None

    pred_ = prediction.T.rename(columns=prediction.T.iloc[0, :]).drop(['ID'])
    df_new_2 = df_new_1.T.rename(columns=df_new_1.T.iloc[0, :]).drop(['ID'])

    for id_ in df_new_2.columns:
        for pred_id in pred_.columns:
            if id_ == pred_id:
                df_new_2[id_] = pred_[id_]

    df_new_3 = df_new_2.T.sort_index()
    print('done')
    return df_new_3

def good(df_list, prediction):
    df_new = np.array(df_list[0]) # 불량에 대한 정보가 들어있는 df_list를 호출하여 np.array로 만들어줌
    for i in range(len(df_list) - 1):
        df_new = np.concatenate([df_new, np.array(df_list[i + 1])])

    df_new_1 = pd.DataFrame(df_new)
    df_new_1 = pd.DataFrame(np.array(df_new_1[~df_new_1.duplicated()]), columns=['ID'])

    for i in df_new_1.index:
        for j in prediction.index:
            if df_new_1.iloc[i, 0] == prediction.iloc[j, 0]:
                prediction = prediction.drop(index=j, axis=0)
    print('done')
    return prediction

def lineplot(prediction, title, mode):
    plt.clf()
    plot = sns.lineplot(data=prediction)
    wandb.log({mode+'_'+title: wandb.Image(plot)})

def histogram(label, prediction, y_feature_path):
    label = label.cpu().detach().numpy()
    prediction = prediction.cpu().detach().numpy()

    label = pd.DataFrame(label)
    prediction = pd.DataFrame(prediction)

    for col in range(len(label.columns)):
        y_feature_info = pd.read_csv(y_feature_path)
        new_result = pd.concat([prediction.iloc[:, col], label.iloc[:, col]], axis=1,
                               keys=["prediction_{}".format(label.columns[col]+int(1)),
                                     "label_{}".format(label.columns[col]+int(1))])
        histplot(new_result, title=y_feature_info.iloc[col, 1])

def histplot(prediction, title):
    plt.clf()
    plot = sns.histplot(data=prediction, bins = 50, element="poly", fill=False) # histogram을 선으로 나타내줌
    # plot = sns.histplot(data=prediction, bins=50) # histogram을 막대그래프로 나타내줌
    # plot = sns.histplot(data=prediction, bins=50, kde=True) # histogram을 막대그래프 + 선으로 나타내줌
    wandb.log({title: wandb.Image(plot)})

def Table(prediction, title):
    outputs_table = wandb.Table(data=prediction, columns=prediction.columns)
    wandb.log({title: outputs_table})

def log(prefix, metrics_dict):
    log_dict = {f'{prefix}_{key}': value for key, value in metrics_dict.items()}
    wandb.log(log_dict)

def plot_decision_regions(x,y,classifier,test_idx=None, resolution=0.2):

    markers = ('x', 's', '0', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=x[y == c1, 0], y=x[y == c1, 1],
                    alpha = 0.8, c=colors[idx],
                    marker=markers[idx], label=c1, edgecolors='black')
    if test_idx:
        x_test, y_test = x[test_idx, :], y[test_idx]
        plt.scatter(x_test[:, 0], y_test[:, 1],
                    facecolors='none', edgecolors='black', alpha=1.0,
                    linewidths=1, marker='o', s=100, label='test set')


