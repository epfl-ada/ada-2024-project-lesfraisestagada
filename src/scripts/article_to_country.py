import sys
sys.path.append('src')

from torch.utils.data import DataLoader
from country_list import countries_for_language
from models.llm_classifier import Generator
from data.dataloader import *
from collections import defaultdict
from tqdm import tqdm
import re
from datasets import Dataset


def country_occurences_in_files():
    """Count the occurences of countries in the plaintext files

    Returns:
        dataframe: A dataframe with the counts of countries in the files
        The format is as follows:
        index : filename,
        columns : countries,
        values : occurences
    """
    countries = list(dict(countries_for_language('en')).values())

    counts = defaultdict(dict)

    for file_ in tqdm(plaintext_files_iterator()):
        filename = os.path.splitext(file_.split('/')[-1])[0]
        with open(file_, 'r', encoding="utf-8") as f:
            content = f.read()
            content = content.lower()
            for country in countries:
                country_pattern = r'\b' + re.escape(country.lower()) + r'\b'
                count = len(re.findall(country_pattern, content)) 
                counts[country.lower()][filename] = count
                
    df = pd.DataFrame(counts)
    return df


def filter_top_k(df, k, N):
    """Filter the top k countries with more than N occurences in the files

    Args:
        df (dataframe): dataframe in the format ouputed by country_occurences_in_files
        k (int): Top k countries per article
        N (int): Minimum occurences of a country to be considered

    Returns:
        dataframe: A dataframe with the top k countries per article
    """
    filtered_df = pd.DataFrame(index=df.index, columns=[f"Top_{i+1}_name" for i in range(k)] + [f"Top_{i+1}_count" for i in range(k)])
    
    for row in df.iterrows():
        filename = row[0]
        data = row[1]
        top_counts = data.sort_values(ascending=False)
        
        top_countries = top_counts[top_counts > N].head(k)
        top_countries_name = top_countries.index.tolist()

        
        for i, country in enumerate(top_countries_name):
            filtered_df.loc[filename, f"Top_{i+1}_name"] = country
            
        for i, country in enumerate(top_countries):
            filtered_df.loc[filename, f"Top_{i+1}_count"] = country
        
    return filtered_df



def count_and_lama(use_counts=True, model_key="meta-llama/Meta-Llama-3.1-8B-Instruct", model_family='llama', file_name='country_occurences'):
    
    df = pd.read_csv('./data/country_data.csv', index_col=0)
    df_counts = filter_top_k(df, k=2, N=1)
    
    if use_counts:
        nan_df = df_counts[df_counts.isna().all(axis=1)]
        print(f"Number of articles with no countries: {len(nan_df)}")
    else:
        nan_df = df_counts
        print(f"Number of articles with no countries: {len(nan_df)}")
    
    countries = list(dict(countries_for_language('en')).values())

        
    generator = Generator(local_compute=False, model_key=model_key, model_family=model_family)
    _, tokenizer = generator.load_model()
    
    system_prompt = f"""You will be given textual articles. For each article provide single and unique country to which the article is related and should be classified to. Provide the answer in the form : <country>.
    If there is no country related to the article, please write 'None'. If the location is not on earth, please write 'None'. You must be 100\% sure this is a question of life.
    This is the list of coutnries that you are allowed to output DON'T output anything that is not in this list: {countries}"""
    
    user_prompt = ""

    inputs = []
    chats = []
    outputs = []
    
    if os.path.exists("data/" + file_name + "_dataset.json"):
        print("Dataset already exists")
        dataset = Dataset.from_json("data/" + file_name + "_dataset.json")
                
        inputs = list(nan_df.index)

    else:
        print("----------------- Applying Chat Template -----------------")
        with tqdm(total=len(nan_df)) as pbar:
            for i, _ in tqdm(nan_df.iterrows()):
                inputs.append(i)
                content = file_finder(article_name=i)
                
                chat = user_prompt + "\n" + content
                chat = generator.chat_to_dict(chat)
                chat = generator.add_system_prompt(system_prompt, chat)
                chat = generator.apply_chat_template(chat, tokenizer)
                
                chats.append({"inputs" : chat})
                pbar.update(1)
            
        dataset = Dataset.from_list(chats)
        print(dataset)
        
        dataset.to_json("data/" + file_name + "_dataset.json")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("----------------- Starting Model Generation -----------------")
    with tqdm(total=len(dataloader)) as pbar:
        for _, batch in enumerate(dataloader):
            output = generator.generate(batch, max_new_tokens=10)
            outputs.extend(output)
            pbar.update(1)
                
    
    if use_counts:
        new_counts = df_counts.copy(deep=True)
        for article, country in zip(inputs, outputs):
            if country != "None":
                
                for country_ in countries:
                    if country_.lower() in country.lower():
                        new_counts.loc[article, "Top_1_name"] = country_.lower()
                        new_counts.loc[article, "Top_1_count"] = 0
                        break
    
    else:
        cleaned_outputs = []
        
        for country in outputs:
            for country_ in countries:
                found = False
                if country_.lower() in country.lower():
                    cleaned_outputs.append(country_.lower())
                    found = True
                    break
            if not found:
                cleaned_outputs.append("")
        
        
        new_counts = pd.DataFrame({"Top_1_name" : cleaned_outputs, "Predictions" : outputs})
        new_counts.index = inputs


    print(f"Number of articles with no countries before completion with llama: {len(nan_df)}")
    nan_df = new_counts[new_counts.isna().all(axis=1)]
    print(f"Number of articles with no countries after completion with llama: {len(nan_df)}")

    new_counts.to_csv("data/" + file_name + ".csv")


if __name__ == "__main__":
    
    count_and_lama()
    
    count_and_lama(use_counts=False, model_key="meta-llama/Meta-Llama-3.1-8B-Instruct", model_family='llama', file_name='country_data_full_llama')
    
    count_and_lama(use_counts=False, model_key="Qwen/Qwen2.5-7B-Instruct", model_family='qwen', file_name='country_data_full_qwen')
    
    count_and_lama(use_counts=False, model_key="google/gemma-2-9b-it", model_family='gemma', file_name='country_data_full_gemma')
    
    