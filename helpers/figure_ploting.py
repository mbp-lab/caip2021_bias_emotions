import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt
import pandas as pd

sns.set(font_scale=1.18)
sns.set_style('ticks')


def modality_changer(x):
    first = 'audio' if x['Voice'] else ''
    second = 'visual' if x['Image'] or x['Video'] else ''
    return first + second


def save_plot(plot, file_name):
    plot = plot.get_figure()
    plot.savefig(f'./plots/{file_name}.pdf')
    plt.clf()


def plot_hist(table, xlabel, file_name):
    plot = sns.histplot(table, bins=20)
    plot.set_xlabel(xlabel)
    # plot.set_title(file_name)
    save_plot(plot, f'{file_name}_hist')


def plot_bar(table, file_name, group_name):
    age_group_names = ['Children', 'Youth', 'Early Adults', 'Late Adults', 'Seniors']
    age_groups = [0, 12, 20, 40, 65, max(74, table[f'{group_name}-age-max'].max()) + 1]
    res_dic = {e: 0 for e in age_group_names}
    for item in table.iterrows():
        min_age, max_age = item[1][f'{group_name}-age-min'], item[1][f'{group_name}-age-max']
        for i, r in enumerate(age_groups):
            if i == 0:
                continue
            if age_groups[i - 1] <= min_age < age_groups[i] or age_groups[i - 1] <= max_age < age_groups[
                i] or min_age < \
                    age_groups[i - 1] < max_age:
                res_dic[age_group_names[i - 1]] += 1
    res_dic = pd.DataFrame(res_dic, index=[0])
    plot = sns.barplot(data=res_dic)
    plot.set_ylabel('Count')
    plot.set_xticklabels(age_group_names)
    save_plot(plot, file_name)


def plot_boxplot(table):
    table['Year'] = table['Year'].apply(lambda x: int((x // 5) * 5))
    table['Modality'] = table.T.apply(modality_changer)

    plot = sns.boxplot(x='Year', y='Samples', hue='Modality',
                       data=table)
    plot.set_yscale('log')
    plot = sns.stripplot(x='Year', y='Samples', hue='Modality',
                         jitter=True,
                         dodge=True,
                         marker='o',
                         palette="Set2",
                         # alpha=0.5,
                         data=table)
    plot.set_yscale('log')
    handles, labels = plot.get_legend_handles_labels()
    plt.legend(handles[0:3], labels[0:3])
    save_plot(plot, 'datasets_size_time_point_5')


def plot_barplot(table):
    table['Year'] = table['Year'].apply(lambda x: int((x // 5) * 5))
    table['Modality'] = table.T.apply(modality_changer)
    table = table.groupby(['Year', 'Modality'], as_index=False)['Dataset'].count()
    a = []
    d = {}
    for y in table['Year'].unique():
        d['Year'] = y
        d['Modality'] = 'visual'
        d['Dataset'] = table[table['Year'] == y][table['Modality'].isin(['visual', 'audio', 'audiovisual'])][
            'Dataset'].sum()
        a.append(d.copy())
        d['Modality'] = 'audio'
        d['Dataset'] = table[table['Year'] == y][table['Modality'].isin(['audio', 'audiovisual'])]['Dataset'].sum()
        a.append(d.copy())
        d['Modality'] = 'audiovisual'
        d['Dataset'] = table[table['Year'] == y][table['Modality'] == 'audiovisual']['Dataset'].sum()
        a.append(d.copy())
    table = pd.DataFrame(a)
    sns.barplot(x='Year', y='Dataset', data=table[table['Modality'] == 'visual'], color=sns.color_palette()[1])
    sns.barplot(x='Year', y='Dataset', data=table[table['Modality'] == 'audio'], color=sns.color_palette()[0])
    plot = sns.barplot(x='Year', y='Dataset', data=table[table['Modality'] == 'audiovisual'],
                       color=sns.color_palette()[2])
    plot.set_ylabel('Count')
    save_plot(plot, 'datasets_count_time')


def plot_boxplot_ethnicity(table, cond):
    sns.boxplot(x="Groups", y="Share in reporting datasets (%)", data=table)
    plot = sns.swarmplot(x="Groups", y="Share in reporting datasets (%)", data=table, color="0", alpha=0.6)
    # plot.plot(plot.get_xlim(), [50] * 2, 'k--', alpha=0.1)
    save_plot(plot, f'{cond}_ethicity')


def plot_raincloud(table, y_col, y_label, file_name):
    table['Year'] = table['Year'].apply(lambda x: int((x // 5) * 5))
    plot = pt.RainCloud(x='Year', y=y_col, data=table, width_viol=1, point_size=6)
    plot.set_ylabel(y_label)
    save_plot(plot, file_name)
