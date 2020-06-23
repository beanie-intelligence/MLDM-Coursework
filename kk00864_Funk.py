import os
from itertools import cycle
import itertools as ite
import random
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
from numpy import interp  # from scipy import interp
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn import svm
from scipy import stats


class Funk:
    def __init__(self):
        self.path_to_data = 'funk_final.csv'
        self.path_to_file = ''
        self.random_seed = 69
        random.seed(self.random_seed)
        self.test_size = 0.1
        self.parameters = {'legend.fontsize': 'x-large',
                           'figure.figsize': (10, 10),
                           'axes.labelsize': 'x-large',
                           'axes.titlesize': 'x-large',
                           'xtick.labelsize': 'x-large',
                           'ytick.labelsize': 'x-large',
                           "lines.linewidth": 2}
        plt.rcParams.update(self.parameters)

    def _load_data(self) -> pd.DataFrame:
        """Loads data into a pandas DataFrame."""
        return pd.read_csv(self.path_to_data)

    def _preprocess_data(self) -> pd.DataFrame:
        """Preprocesses the data into a pandas DataFrame."""
        df = self._load_data()
        df = df.drop_duplicates('track_name').sort_index()
        return df[['track_id', 'track_name', 'artist_name', 'album_name', 'time_signature', 'duration_ms', 'acousticness', 'energy', 'key', 'liveness', 'loudness', 'mode', 'popularity',
                   'speechiness', 'tempo', 'valence', 'danceability']]

    def _hist_plots(self):
        """Plots histogram of data and returns a description of the data."""
        df = self._preprocess_data()
        hists = df.hist(figsize=(10, 10))
        plt.show()
        pd.set_option('precision', 3)
        return df.describe().T[['mean', 'std', 'max', 'min', '25%', '75%']]

    def _skewness(self):
        df = self._preprocess_data()
        """Computes the skewness of the data distribution."""
        return df.skew()

    def _skew_plots(self):
        def _skew_per(col: str):
            # Removing the skew by using the boxcox transformations
            df = self._preprocess_data()
            tf = np.asarray(df[[col]].values.tolist()).ravel()
            df_tf = stats.boxcox(tf)[0]
            tf1 = np.asarray(df[[col]].values.tolist()).ravel()
            df_tf1 = stats.boxcox(tf1)[0]
            sns.distplot(df[col], bins=20, kde=True, kde_kws={
                "color": "k", "label": f"Distribution pre-skew\n{col}"}, color='yellow')
            plt.savefig(
                self.path_to_file + f'Figures/Funk-distribution-hist-{col}-pre-skew.png')
            sns.distplot(df_tf1, bins=20, kde=True, kde_kws={
                "color": "b", "label": f"Distribution after-skew\n{col}"}, color='black')  # corrected skew data
            plt.savefig(
                self.path_to_file + f'Figures/Funk-distribution-hist-{col}-after-skew.png')
            plt.show()

        for i in ['acousticness', 'energy', 'liveness', 'popularity',
                  'speechiness', 'tempo', 'valence', 'danceability']:
            _skew_per(col=str(i))

    def _correlation(self):
        """Plots and prints the correlation between the selected variables."""
        df = self._preprocess_data()
        corr = df.corr(method='pearson')
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        fig, ax = plt.subplots()
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, linewidths=.5, cmap=cmap, center=0, ax=ax)
        ax.set_title('Correlation')
        plt.savefig(self.path_to_file + f'Figures/Funk-Correlation.png')
        plt.show()
        corr_metrics = df.corr()
        corr_metrics.style.background_gradient()
        print('The correlation between:\n')
        drop_potential, combinations_to_keep = [], []
        for i in corr:
            for c, j in enumerate(corr[i].values):
                if (j >= 0.25) and (j <= 0.99):
                    print('{} \t\t {:.3f} \t\t {}'.format(corr.index[c], j, i))
                    combinations_to_keep.append([corr.index[c], i])
                elif (j >= 0.99):
                    drop_potential.append([corr.index[c], j, i])

    def relationship_exploration(self):
        """Plotting the relationship between energy and loudness. 
        Then plotting the relationship between danceability and selected features."""
        df = self._preprocess_data()
        fig = plt.subplots()
        sns.regplot(x='energy', y='loudness', data=df, color='black')
        plt.savefig(self.path_to_file + f'Figures/Funk-loudness-energy.png')
        plt.show()
        items = ['acousticness', 'energy', 'liveness', 'popularity',
                 'speechiness', 'tempo', 'valence', 'danceability']
        for i in items[:-1]:
            fig = plt.subplots()
            sns.regplot(x='danceability', y=i, data=df, color='black')
            plt.savefig(self.path_to_file +
                        f'Figures/Funk-danceability-{i}.png')
            plt.show()

    @staticmethod
    def _create_danceability_label(df: pd.DataFrame):
        """Creates the danceability labels."""
        dance_lab = [0, 1, 2, 3]
        s1, s2, s3, s4 = [], [], [], []
        s = []
        for c, i in enumerate(df.danceability):
            if i >= 0.00 and i < 0.50:
                s1.append(i)
                s.append(dance_lab[0])
            elif i >= 0.50 and i < 0.70:
                s2.append(i)
                s.append(dance_lab[1])
            elif i >= 0.70 and i < 0.80:
                s3.append(i)
                s.append(dance_lab[2])
            elif i >= 0.80 and i <= 1.00:
                s4.append(i)
                s.append(dance_lab[3])
        print('0: Not danceable: {}\n1: Somewhat Danceable {}\n2: Very Danceable {}\n3: Extremely Danceable {}'.format(
            len(s1), len(s2), len(s3), len(s4)))
        df["danceability_labels"] = s
        df['danceability_labels'] = df['danceability_labels'].astype('int')

    def _split_data(self):
        """Splits the data into features (X) and labels (y)."""
        df = self._preprocess_data()
        self._create_danceability_label(df=df)
        X = df[['energy', 'key', 'valence', 'tempo', 'mode']]
        y = df['danceability_labels']
        return (X, y)

    def _confusion_matrix_plt(self, matrix, title):
        """Plots confusion matrix."""
        sns.set(style='whitegrid')
        fig = plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(1, 2, 1)
        ax1 = sns.heatmap(matrix, cmap="Blues", annot=True, fmt='')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        plt.savefig(self.path_to_file +
                    f'Figures/Funk-confusion-matrix-{title}')
        plt.show()

    def model1(self):
        """SVM model."""
        X, y = self._split_data()
        # Binarize the output
        y = label_binarize(y, classes=[0, 1, 2, 3])
        n_classes = y.shape[1]
        # shuffle and split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_seed)
        # Learn to predict each class against the other
        classifier = OneVsRestClassifier(svm.SVC(
            kernel='poly', degree=2, probability=True, tol=1e-6, random_state=self.random_seed))  # , gamma= 0.1))
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
        # Binarize the output
        y = label_binarize(y, classes=[0, 1, 2, 3])
        n_classes = y.shape[1]
        # shuffle and split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_seed)
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

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

        colors = cycle(['cyan', 'magenta', 'yellow', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig(
            self.path_to_file + f'Figures/Funk-OneVsRestClassifier-SVM-poly-ROC_.png')
        plt.show()

        y_prob = classifier.predict_proba(X_test)

        macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
                                          average="macro")
        weighted_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
                                             average="weighted")
        macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                          average="macro")
        weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                             average="weighted")
        print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
              "(weighted by prevalence)"
              .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
        print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
              "(weighted by prevalence)"
              .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))

    def model2(self):
        """Decision Tree model."""
        X, y = self._split_data()
        # Decision Tree (Attempt #2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed)

        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        cm_tree = confusion_matrix(y_test, y_pred)
        print(cm_tree)
        print(classification_report(y_test, y_pred))
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(
            metrics.mean_squared_error(y_test, y_pred)))
        cm_tree_df = pd.DataFrame(cm_tree)
        self._confusion_matrix_plt(
            cm_tree_df, title='Decision Tree')

    def model3(self):
        """Bag of models with kfold validation."""
        X, y = self._split_data()
        X_train, X_validation, Y_train, Y_validation = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed)

        # Spot Check Algorithms
        models = []
        models.append(('Naive Bayes', OneVsRestClassifier(GaussianNB())))
        models.append(('KNN', OneVsRestClassifier(KNeighborsClassifier(n_neighbors=len(
            X.columns)+1, metric='euclidean', n_jobs=-1, weights='distance'))))
        extra_tree = ExtraTreeClassifier(random_state=self.random_seed)
        models.append(('Bagging', OneVsRestClassifier(
            BaggingClassifier(extra_tree, random_state=self.random_seed))))
        models.append(
            ('Decision Tree', OneVsRestClassifier(DecisionTreeClassifier())))

        # evaluate each model in turn
        results = []
        names = []
        no_splits = 10

        for c, (name, model) in enumerate(models):
            kfold = StratifiedKFold(
                n_splits=no_splits, random_state=self.random_seed)
            cv_results = cross_val_score(
                model, X_train, Y_train, cv=kfold, scoring='accuracy')
            results.append(cv_results)
            names.append(name)

            print('{}: {} ({})'.format(
                name, cv_results.mean(), cv_results.std()))

        # Compare Algorithms
        plt.boxplot(results, labels=names)
        plt.legend([f"{names[i]} : {round(results[i].mean(),3)}" for i in range(
            len(names))], loc='upper right')
        plt.savefig(
            self.path_to_file + f'Figures/Funk-{no_splits}-fold cross-validation.png')
        plt.show()

        # Binarize the data
        y = label_binarize(y, classes=[0, 1, 2, 3])
        n_classes = y.shape[1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            random_state=self.random_seed)

        # ROC
        for c, (name, model) in enumerate(models):
            y_score = model.fit(X_train, y_train).predict(X_test)
            # Compute ROC curve and ROC area for each class
            fpr,  tpr, roc_auc = dict(), dict(), dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(
                y_test.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate(
                [fpr[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

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

            colors = cycle(['cyan', 'magenta', 'yellow', 'cornflowerblue'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                         ''.format(i, roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.savefig(
                self.path_to_file + f'Figures/Funk-OneVsRestClassifier-SVM-poly-ROC_{name}.png')
            plt.show()
            y_prob = model.predict_proba(X_test)
            macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
                                              average="macro")
            weighted_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
                                                 average="weighted")
            macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                              average="macro")
            weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                                 average="weighted")
            print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
                  "(weighted by prevalence)"
                  .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
            print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
                  "(weighted by prevalence)"
                  .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))
