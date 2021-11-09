import pandas as pd
from pandas.core.reshape.merge import merge

# Initialize a list to use to build a dataframe
# name_of_database = [(input csv column, the output csv column)]
pubmed = [("Title", "Article title"),
          (None, "Abstract"),
          ("DOI", "DOI"),
          ("Publication Year", "Publication year"),
          ("Journal/Book", "Journal"),
          (None, "keywords"),
          (None, "Databases the paper appears in "),
          (None, "N of databases the paper appears in"),
          (None, "Relevance "),
          (None, "Alternative rating"),
          (None, "Use or dataset"),
          (None, "Notes")]

wos = [("Article Title", "Article title"),
       ("Abstract", "Abstract"),
       ("DOI", "DOI"),
       ("Publication Year", "Publication year"),
       ("Journal Abbreviation", "Journal"),
       ("Author Keywords", "keywords"),
       (None, "Databases the paper appears in "),
       (None, "N of databases the paper appears in"),
       (None, "Relevance "),
       (None, "Alternative rating"),
       (None, "Use or dataset"),
       (None, "Notes")]


def create_dict(list_database):
    """Create a dict from the database_key"""
    return_dict = dict()
    for key, value in list_database:
        if key != None:
            return_dict[key] = value
    return return_dict


def create_database_dataframe(dataframe, database_key):
    # create a dataframe with the required columns
    df = pd.DataFrame(dataframe, columns=[column[0] for column in database_key if column[0] != None])

    # convert the the column names
    conversion_dict = create_dict(database_key)
    df.rename(columns=conversion_dict, inplace=True)

    # add missing columns
    column_list = [column[1] for column in database_key]
    return df.reindex(columns=column_list)


def extract_abstract(location):
    with open(location) as f:
        lines = f.read()
        abstract_collection = {}
        for i, line in enumerate(lines.split("\n\n\n")):
            flag = False
            abstract = []
            for text_1 in line.split("\n\n"):
                abstract.append(text_1)
                if "Author information" in text_1:
                    flag = True
            if not flag:
                abstract.insert(0, None)
            abstract_collection[i] = abstract[4]
    return pd.DataFrame.from_dict(abstract_collection, orient="index")


def extract_files(files):
    list_of_all_database = []

    for file, abstract_file in files:
        # parse different databases
        if "wos" in file:
            df = pd.read_excel(file)
            df = create_database_dataframe(df, wos)
            df["Databases the paper appears in "] = "wos"
            list_of_all_database.append(df)

        elif "pub" in file:
            abstract_df = extract_abstract(abstract_file)
            df = pd.read_csv(file)
            df = create_database_dataframe(df, pubmed)
            df["Databases the paper appears in "] = "pubMed"
            df["Abstract"] = abstract_df
            df.to_csv("./abstact_pubmed.csv")
            list_of_all_database.append(df)
        # add more elif conditions as needed

    return list_of_all_database


def merge_files(database_list):
    # merge all the dataframes
    merged_df = pd.concat([df for df in database_list], ignore_index=True)
    cmnts = {}
    # thanks to stackoverflow https://stackoverflow.com/questions/36271413/pandas-merge-nearly-duplicate-rows-based-on-column-value ;)
    # filters out duplicate rows
    for i, row in merged_df.iterrows():
        while True:
            try:
                if row['Databases the paper appears in ']:
                    cmnts[row['Article title']].append(row['Databases the paper appears in '])

                else:
                    cmnts[row['Article title']].append('n/a')

                break

            except KeyError:
                cmnts[row['Article title']] = []

    merged_df.drop_duplicates('Article title', inplace=True)
    merged_df['Databases the paper appears in '] = [';'.join(v) for v in cmnts.values()]
    merged_df["N of databases the paper appears in"] = [len(v) for v in cmnts.values()]
    return merged_df


if __name__ == "__main__":
    file_location = ["./files/dataset_papers/wos.xls",
                     './files/pubmed.csv']
    abstract_location = [None, "./files/pubmed_abstract.txt"]
    files = [(file_location[i], abstract_location[i]) for i in range(len(abstract_location))]
    database_list = extract_files(files)
    df = merge_files(database_list)
    df.to_excel("./files/combined_review_papers_wos_pubmed.xls")
    print('Done')
