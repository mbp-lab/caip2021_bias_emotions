import os
import pandas as pd
import scipy.stats as stats
from pathlib import Path
from datetime import datetime
from helpers.figure_ploting import plot_bar, plot_hist, plot_boxplot, plot_barplot, plot_raincloud

file = './files/extracted_datasets.xlsx'


def is_normally_distributed(dist):
    k2, p = stats.normaltest(dist)
    alpha = 1e-3
    print("k2 = {:g} p = {:g}".format(k2, p))
    return p > alpha


def chi_test(table):
    stat, p, dof, expected = stats.chi2_contingency(table)
    prob = 0.95
    critical = stats.chi2.ppf(prob, dof)
    print('dof=%d' % dof)
    print(expected)
    print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if abs(stat) >= critical:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')


class DatasetsAnalyzer:
    def __init__(self, data):
        self.data = data
        self.stats = {'Date': str(datetime.now())}

    def analyze(self):
        self.calculate_general()
        self.calculate_gender()
        self.calculate_age()
        self.calculate_ethnicity()

        self.analyze_general()
        self.analyze_gender()
        self.analyze_age()
        self.analyze_ethnicity()

        self.plot_general()
        self.plot_gender()
        self.plot_age()
        self.plot_ethnicity()

        self.save_stats()

        self.data.to_excel('./files/result_sheet.xlsx')

    def save_stats(self):
        stats_file_path = Path('./files/stats_sheet.xlsx')
        if stats_file_path.exists():
            data = pd.read_excel(stats_file_path)
        else:
            data = pd.DataFrame(columns=self.stats.keys())
        data = data.append(self.stats, ignore_index=True).reset_index()
        data.to_excel(stats_file_path)

    def calculate_general(self):
        pass

    def calculate_age(self):
        def get_min_max(age):
            age = str(age)
            index = age.find('-')
            age_min_max = (int(age[:index]), int(age[index + 1:])) if index > -1 else (None, None)
            return age_min_max

        age_min_max = self.data['Age range'].apply(get_min_max)
        self.data['P-age-min'] = age_min_max.apply(lambda x: x[0])
        self.data['P-age-max'] = age_min_max.apply(lambda x: x[1])
        self.data['P-age-span'] = age_min_max.apply(lambda x: int(x[1] - x[0]) if x[0] else None)

        r_age_min_max = self.data['ER-age range'].apply(get_min_max)
        self.data['R-age-min'] = r_age_min_max.apply(lambda x: x[0])
        self.data['R-age-max'] = r_age_min_max.apply(lambda x: x[1])
        self.data['R-age-span'] = r_age_min_max.apply(lambda x: int(x[1] - x[0]) if x[0] else None)

    def calculate_gender(self):
        self.data['P-male ratio (m/(f+m))'] = self.data['N-male'] / (self.data['N-female'] + self.data['N-male'])
        self.data['R-male ratio (m/(f+m))'] = self.data['N-ER-male'] / (
                self.data['N-ER-female'] + self.data['N-ER-male'])

    def calculate_ethnicity(self):
        pass

    def analyze_general(self):
        self.stats['Total datasets'] = len(self.data)
        print(f'Count of voice datasets: {self.data["Voice"].sum()}')
        print(f'Count of image datasets: {self.data["Image"].sum()}')
        print(f'Count of video datasets: {self.data["Video"].sum()}')

        def print_mean_std_median(column_name):
            print(
                f'Mean: P{column_name}: {self.data[f"P{column_name}"].mean()} R{column_name}: {self.data[f"R{column_name}"].mean()}')
            print(
                f'Std: P{column_name}: {self.data[f"P{column_name}"].std()} R{column_name}: {self.data[f"R{column_name}"].std()}')
            print(
                f'Median: P{column_name}: {self.data[f"P{column_name}"].median()} R{column_name}: {self.data[f"R{column_name}"].median()}')

        print(self.data.describe())

        print_mean_std_median('-male ratio (m/(f+m))')
        print_mean_std_median('-age-min')
        print_mean_std_median('-age-max')
        print_mean_std_median('-age-span')

    def analyze_age(self):
        table = self.data[['Year', 'P-age-span']][self.data['P-age-span'].notna()][self.data['Year'].notna()]
        self.stats['Normality p-age'] = str(stats.normaltest(table))
        self.stats['Pearson p-age'] = str(stats.pearsonr(table['Year'].values, table['P-age-span'].values))
        self.stats['Spearman p-age'] = str(stats.spearmanr(table['Year'].values, table['P-age-span'].values))
        table = self.data[['Year', 'R-age-span']][self.data['R-age-span'].notna()][self.data['Year'].notna()]
        self.stats['Normality r-age'] = str(stats.normaltest(table))
        self.stats['Pearson r-age'] = str(stats.pearsonr(table['Year'].values, table['R-age-span'].values))
        self.stats['Spearman r-age'] = str(stats.spearmanr(table['Year'].values, table['R-age-span'].values))

    def analyze_gender(self):
        table = self.data['P-male ratio (m/(f+m))'][self.data['P-male ratio (m/(f+m))'].notna()]
        self.stats['Normality p-gender'] = str(stats.normaltest(table))
        self.stats['TTest p-gender'] = str(stats.ttest_1samp(a=table, popmean=0.5))
        self.stats['Wilcoxon p-gender'] = str(stats.wilcoxon(table - 0.5))
        table = self.data['R-male ratio (m/(f+m))'][self.data['R-male ratio (m/(f+m))'].notna()]
        self.stats['Normality r-gender'] = str(stats.normaltest(table))
        self.stats['TTest r-gender'] = str(stats.ttest_1samp(a=table, popmean=0.5))
        self.stats['Wilcoxon r-gender'] = str(stats.wilcoxon(table - 0.5))

        # print('Chi-squared test for participants')
        # table = self.data[['N-male', 'N-female']][self.data['N-male'].notna()].T
        # chi_test(table)
        # # self.stats['Chi-squared p-gender'] =
        # print('Chi-squared test for raters')
        # table = self.data[['N-ER-male', 'N-ER-female']][self.data['N-ER-male'].notna()].T
        # chi_test(table)

        table = self.data[['Year', 'P-male ratio (m/(f+m))']][self.data['P-male ratio (m/(f+m))'].notna()][
            self.data['Year'].notna()]
        self.stats['Pearson p-gender'] = str(
            stats.pearsonr(table['Year'].values, table['P-male ratio (m/(f+m))'].values))
        self.stats['Spearman p-gender'] = str(
            stats.spearmanr(table['Year'].values, table['P-male ratio (m/(f+m))'].values))
        table = self.data[['Year', 'R-male ratio (m/(f+m))']][self.data['R-male ratio (m/(f+m))'].notna()][
            self.data['Year'].notna()]
        self.stats['Pearson r-gender'] = str(
            stats.pearsonr(table['Year'].values, table['R-male ratio (m/(f+m))'].values))
        self.stats['Spearman r-gender'] = str(
            stats.spearmanr(table['Year'].values, table['R-male ratio (m/(f+m))'].values))

    def analyze_ethnicity(self):
        pass

    def plot_general(self):
        table = self.data[['Year', 'Samples', 'Voice', 'Image', 'Video']][self.data['Samples'].notna()][
            self.data['Year'].notna()]
        plot_boxplot(table)

        table = self.data[['Year', 'Dataset', 'Voice', 'Image', 'Video']][self.data['Year'].notna()]
        plot_barplot(table)

    def plot_age(self):
        table = self.data[['P-age-min', 'P-age-max']][self.data['P-age-min'].notna()]
        plot_bar(table, 'p-age-ranges_bins', 'P')

        table = self.data[['R-age-min', 'R-age-max']][self.data['R-age-min'].notna()]
        plot_bar(table, 'r-age-ranges_bins', 'R')

        table = self.data[['Year', 'P-age-span']][self.data['P-age-span'].notna()][self.data['Year'].notna()]
        plot_raincloud(table, 'P-age-span', 'Age span', 'datasets_span_time_rain_5')

    def plot_gender(self):
        table = self.data['P-male ratio (m/(f+m))'][self.data['N-male'].notna()]
        plot_hist(table, 'Male ratio (m/(f+m))', 'participants_cumulative_hist')

        table = self.data['R-male ratio (m/(f+m))'][self.data['N-ER-male'].notna()]
        plot_hist(table, 'Male ratio (m/(f+m))', 'raters_cumulative_hist')

        table = self.data[['Year', 'P-male ratio (m/(f+m))']][self.data['P-male ratio (m/(f+m))'].notna()][
            self.data['Year'].notna()]
        plot_raincloud(table, 'P-male ratio (m/(f+m))', 'Male ratio (m/(f+m))', 'datasets_gender_time_rain_5')

    def plot_ethnicity(self):
        pass


if __name__ == '__main__':
    df = pd.read_excel(file)
    analyzer = DatasetsAnalyzer(df)
    analyzer.analyze()
