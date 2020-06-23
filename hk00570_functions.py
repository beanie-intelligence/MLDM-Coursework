import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import minisom
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from itertools import cycle
from sklearn.multiclass import OneVsRestClassifier

pd.set_option('display.max_columns', 999)


class HKClassifierClass:

    def __init__(self, target):
        self.data_orig = pd.read_csv(filepath_or_buffer='spotify_data_genre_playlists.csv')
        self.data = self.data_orig[['track_id', 'track_name', 'artist_name', 'album_name', 'genre',
                                    'popularity', 'danceability', 'energy', 'loudness', 'speechiness',
                                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
        self.feature_names = self.data.columns
        self.target = target

    def explore_data(self):
        sns.set(color_codes=True)
        plt.figure(figsize=[12.8, 15])
        for c, i in enumerate(['danceability', 'energy', 'loudness', 'speechiness',
                               'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']):
            plt.subplot(331 + c)
            ax = sns.distplot(self.data[i])
            ax.set_title('Distribution ' + i)
            ax.set_ylabel('%')
            ax.set_xlabel('Value')
        plt.show()

        data_cols = self.data[['danceability', 'energy', 'loudness', 'speechiness',
                               'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
        corr = data_cols.corr()
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(corr, vmax=1.0, vmin=-1.0, center=0, fmt='.2f', cmap='coolwarm',
                    square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
        plt.show()

    def pre_process_data(self):
        self.data['loudness'] = (1 - (self.data['loudness'] / (-60)))
        self.data['tempo'] = self.data['tempo'] / 250
        self.data['genre'] = np.where((self.data['genre'] == 'country'), 0,
                             np.where((self.data['genre'] == 'funk'), 1,
                             np.where((self.data['genre'] == 'hiphop'), 2,
                             np.where((self.data['genre'] == 'jazz'), 3,
                             np.where((self.data['genre'] == 'metal'), 4,
                             np.where((self.data['genre'] == 'pop'), 5,
                             np.where((self.data['genre'] == 'reggae'), 6,
                             np.where((self.data['genre'] == 'rock'), 7,
                             np.where((self.data['genre'] == 'soul'), 8, -9999)))))))))
        self.data['popularity'] = np.where((self.data['popularity'] >= 0) & (self.data['popularity'] < 20), 0,
                                  np.where((self.data['popularity'] >= 20) & (self.data['popularity'] < 40), 1,
                                  np.where((self.data['popularity'] >= 40) & (self.data['popularity'] < 60), 2,
                                  np.where((self.data['popularity'] >= 60) & (self.data['popularity'] < 80), 3,
                                  np.where((self.data['popularity'] >= 80), 4, -9999)))))
        self.data =  self.data[['genre', 'popularity', 'danceability', 'loudness', 'speechiness',
                                'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
        self.feature_names = ['danceability', 'loudness', 'speechiness', 'acousticness',
                              'instrumentalness', 'liveness', 'valence', 'tempo']

    def classify_som(self, som_model, data, class_assignments):
        winmap = class_assignments
        default_class = np.sum(list(winmap.values())).most_common()[0][0]
        result = []
        for d in data:
            win_position = som_model.winner(d)
            if win_position in winmap:
                result.append(winmap[win_position].most_common()[0][0])
            else:
                result.append(default_class)
        return result

    def explore_som_classification_parameters(self):
        labels = self.data[self.target]
        features = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, self.data[self.feature_names])
        X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels)
        results = []
        for lr in [0.1, 0.01, 0.001]:
            for sig in [5, 10, 15]:
                for fct in ['gaussian', 'mexican_hat', 'bubble', 'triangle']:
                    som = minisom.MiniSom(50, 50, 8, sigma=sig, learning_rate=lr, neighborhood_function=fct,
                                          random_seed=10)
                    som.train(X_train, 1000, verbose=False)
                    class_assignment = som.labels_map(X_train, y_train)
                    y_pred = self.classify_som(som, X_test, class_assignment)
                    results.append([lr, sig, fct, sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True)])
        names = ['learning rate', 'sigma', 'neighborhood fct.', 'accuracy score']
        plt.figure(figsize=[20, 20])
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.axis('tight')
        df = pd.DataFrame(np.array(results), columns=names)
        ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        plt.show()

    def execute_som_model(self, size, iterations, sigma, learning_rate, neighborhood_function):
        labels = self.data[self.target]
        features = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, self.data[self.feature_names])
        X_train, X_test, y_train, y_test = train_test_split(features, labels, stratify=labels)
        final_som = minisom.MiniSom(size, size, 8, sigma=sigma,
                                    learning_rate=learning_rate,
                                    neighborhood_function=neighborhood_function)
        final_som.train(X_train, iterations, verbose=False)
        class_assignment = final_som.labels_map(X_train, y_train)
        y_pred = self.classify_som(final_som, X_test, class_assignment)

        print('METRICS FOR THE SOM with size: ' + str(size) + ', ' + str(iterations) +
              ' iterations, sigma ' + str(sigma) + ', learning rate: ' + str(learning_rate) +
              ' and neighborhood function: ' + str(neighborhood_function))
        w = final_som.get_weights()
        plt.figure(figsize=[12.8, 15])
        for c, i in enumerate(['danceability', 'loudness', 'speechiness', 'acousticness',
                               'instrumentalness', 'liveness', 'valence', 'tempo']):
            plt.subplot(421 + c)
            plt.title(i)
            plt.pcolor(w[:, :, c].T, cmap='Spectral')
            plt.xticks(np.arange(size + 1, step=size/6))
            plt.yticks(np.arange(size + 1, step=size/6))
        plt.tight_layout()
        plt.show()
        print('Accuracy Score: ' + str(sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True)))
        print(sklearn.metrics.confusion_matrix(y_test, y_pred))
        print(sklearn.metrics.classification_report(y_test, y_pred))

    def som_10_fold_cross_validation(self, size, iterations):
        data = self.data[[self.target, 'danceability', 'loudness', 'speechiness', 'acousticness',
                          'instrumentalness', 'liveness', 'valence', 'tempo']]
        data = data.sample(frac=1)
        spl = 10

        n_rows = int(data.shape[0])
        a = int(np.floor(n_rows / spl))

        end = []

        for i in range(spl - 1):
            end.append(data.iloc[a * i: a * (i + 1)])
        end.append(data.iloc[(8 * a):(n_rows + 1)])

        acc = []

        for i in end:
            test_frame = i
            train_frame = pd.concat([x for x in end if not x.equals(i)])

            y_test = (test_frame[self.target]).to_numpy()
            y_train = (train_frame[self.target]).to_numpy()

            X_test = (test_frame[['danceability', 'loudness', 'speechiness', 'acousticness',
                                  'instrumentalness', 'liveness', 'valence', 'tempo']]).to_numpy()
            X_train = (train_frame[['danceability', 'loudness', 'speechiness', 'acousticness',
                                    'instrumentalness', 'liveness', 'valence', 'tempo']]).to_numpy()

            final_som = minisom.MiniSom(size, size, 8, sigma=5, learning_rate=0.01, neighborhood_function='triangle')
            final_som.train(X_train, iterations, verbose=False)

            class_assignment = final_som.labels_map(X_train, y_train)
            y_pred = self.classify_som(final_som, X_test, class_assignment)
            acc.append(np.average(sklearn.metrics.accuracy_score(y_test, y_pred, normalize=True)))
        print('AVERAGE 10-FOLD CROSS VALIDATION ACCURACY: ')
        print(np.average(acc))

    def execute_tree_model(self):
        labels = self.data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(self.data[self.feature_names], labels, stratify=labels)

        tree = DecisionTreeClassifier(random_state=0, criterion='entropy').fit(X_train, y_train)
        plt.figure(figsize=[12.8, 15])
        plot_tree(tree, filled=True)
        plt.show()

        y_pred = tree.predict(X_test)

        print('DECISION TREE DEFAULT:')
        print('Accuracy Score on Test: ' + str(sklearn.metrics.accuracy_score(y_test, y_pred)))
        print('Confusrion Matrix: ' + str(sklearn.metrics.confusion_matrix(y_test, y_pred)))
        print('Classification Report: ' + str(sklearn.metrics.classification_report(y_test, y_pred)))
        return tree

    def explore_tree_pruning(self):
        labels = self.data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(self.data[self.feature_names], labels, stratify=labels)
        tree_prune = DecisionTreeClassifier(random_state=0, criterion='entropy')
        alphas = tree_prune.cost_complexity_pruning_path(X_train, y_train).ccp_alphas
        trees = []
        for alpha in alphas:
            trees.append(
                DecisionTreeClassifier(random_state=0, criterion='entropy', ccp_alpha=alpha).fit(X_train, y_train))

        train_scores = [i.score(X_train, y_train) for i in trees]
        test_scores = [i.score(X_test, y_test) for i in trees]

        fig, ax = plt.subplots()
        ax.set_xlabel('alpha')
        ax.set_ylabel('accuracy')
        ax.set_title('accuracy vs. alpha')
        ax.plot(alphas, train_scores, marker='o', label='train', drawstyle='steps-post')
        ax.plot(alphas, test_scores, marker='o', label='test', drawstyle='steps-post')
        ax.legend()
        plt.show()

    def tree_10_fold_cross_validation(self, tree_model):
        data = self.data[[self.target, 'danceability', 'loudness', 'speechiness', 'acousticness',
                          'instrumentalness', 'liveness', 'valence', 'tempo']]
        labels = data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(data[self.feature_names], labels, stratify=labels)
        kfold = sklearn.model_selection.StratifiedKFold(n_splits=10, random_state=1)
        print('AVERAGE 10-FOLD CROSS VALIDATION ACCURACY: ')
        print(sklearn.model_selection.cross_val_score(tree_model,X_train, y_train, cv=kfold, scoring='accuracy'))

    def plot_ROC_curve(self):
        labels = np.array(self.data[self.target])
        classes = [0,1,2,3,4] if self.target == 'popularity' else [0,1,2,3,4,5,6,7,8]
        labels = sklearn.preprocessing.label_binarize(labels, classes=classes)
        n_classes = labels.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(self.data[self.feature_names], labels, test_size=0.15, random_state=69)

        # Learn to predict each class against the other
        classifier = OneVsRestClassifier(DecisionTreeClassifier())
        y_score = classifier.fit(X_train, y_train).predict(X_test)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test[:,i], y_score[:,i])
            roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'yellow', 'green', 'yellow', 'pink', 'magenta']) if self.target == 'genre' else cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'yellow'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()
