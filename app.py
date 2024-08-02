from flask import Flask, render_template, request
import pickle
import numpy as np
from fuzzywuzzy import process
import string
app = Flask(__name__)

def recommend(item_name, similarity_score, jaccard_similarity, pearson_similarity, data_rec, k):
    items = set()  # Use a set to avoid duplicate recommendations
    found = True
    similarity_score = np.array(similarity_score)
    jaccard_similarity = np.array(jaccard_similarity)
    pearson_similarity = np.array(pearson_similarity)

    if item_name not in data_rec.index:
        print(f'Item "{item_name}" not found in the dataset.')
        found = False
        return [], found

    index = np.where(data_rec.index == item_name)[0]
    if len(index) == 0:
        print(f'No similar items found for item "{item_name}".')
        return [], False

    index = index[0]
    similar_item_1 = sorted(enumerate(similarity_score[index]), key=lambda x: x[1], reverse=True)[1:k]
    similar_item_2 = sorted(enumerate(jaccard_similarity[index]), key=lambda x: x[1], reverse=True)[1:k]
    similar_item_3 = sorted(enumerate(pearson_similarity[index]), key=lambda x: x[1], reverse=True)[1:k]

    for i in similar_item_1:
        items.add(data_rec.index[i[0]])
    for i in similar_item_2:
        items.add(data_rec.index[i[0]])
    for i in similar_item_3:
        items.add(data_rec.index[i[0]])

    items = list(items)
    items.sort(key=lambda x: (similarity_score[np.where(data_rec.index == x)[0][0]].mean(),
                              jaccard_similarity[np.where(data_rec.index == x)[0][0]].mean(),
                              pearson_similarity[np.where(data_rec.index == x)[0][0]].mean()), reverse=True)

    return items, found

def find_nearest_item_name(item_list, input_item, threshold = 85):
    input_item = input_item.lower().strip()
    normalized_item_list = [item.lower().strip() for item in item_list]
    match, score = process.extractOne(input_item, normalized_item_list)
    print(score)
    if score >= threshold:
        return item_list[normalized_item_list.index(match)]
    else:
        return None

file_path1 = r"C:\Users\VICTUS\Documents\GitHub\ZEPTO1\data_rec.pkl"
file_path2 = r"C:\Users\VICTUS\Documents\GitHub\ZEPTO1\jaccard_similarity.pkl"
file_path3 = r"C:\Users\VICTUS\Documents\GitHub\ZEPTO1\pearson_similarity.pkl"
file_path4 = r"C:\Users\VICTUS\Documents\GitHub\ZEPTO1\similarity_score.pkl"
file_path5 = r"C:\Users\VICTUS\Documents\GitHub\ZEPTO1\top_20_product.pkl"
file_path6 = r"C:\Users\VICTUS\Documents\GitHub\ZEPTO1\product_name.pkl"

data_rec = pickle.load(open(file_path1, 'rb'))
jaccard_similarity = pickle.load(open(file_path2, 'rb'))
pearson_similarity = pickle.load(open(file_path3, 'rb'))
similarity_score = pickle.load(open(file_path4, 'rb'))
top_20 = pickle.load(open(file_path5, 'rb'))
product_name = pickle.load(open(file_path6,'rb'))

@app.route('/')
def welcome():
    item_name = list(top_20[0].values)
    image_url = list(top_20[1].values)
    return render_template('index.html',
                           item_name=item_name,
                           image_url=image_url,
                           zip=zip)

@app.route('/recommender', methods=['GET', 'POST'])
def recommender():
    recommended_items = []  # Initialize the variable
    product_names = list(product_name[0].values)
    if request.method == 'POST':
        item = request.form['recommended_items']
        item = find_nearest_item_name(product_names,item)
        recommended_items, found = recommend(item, similarity_score, jaccard_similarity, pearson_similarity, data_rec, 10)

    return render_template('recommender.html', recommended_items=recommended_items)

if __name__ == '__main__':
    app.run(debug=True)
