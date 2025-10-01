import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
from collections import defaultdict
import os
import re
import math
import numpy as np
from collections import defaultdict
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle



# Download necessary NLTK packages

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)






#we first load the stop words from teh stop word filw 

stopwords = set()
with open("Stopword-List.txt", 'r') as f:
    for line in f:
        word = line.strip()
        if word:  # Ignore empty lines
            stopwords.add(word)

# we will now check if index file exists, load it if it does, if not  extract the text from the given documents in the abstract 
# convert the text into tokens and store the text the tokens in that document and the total unioque words in all documents
# function to load previously created iindex

def checkindex():
    index_file = "index.pkl"

    if os.path.exists(index_file):
        with open(index_file, 'rb') as f:
            data = pickle.load(f)
        return data['documents'], data['documenttok'], data['words'], data['tf'], data['df'], data['idf'], data['tfidf'], data['documentvector']
    
    return None  


lemmatizer = WordNetLemmatizer()
loaded =checkindex()


if loaded:
    documents, documenttok, words, tf, df, idf, tfidf, documentvector = loaded
    
    
else:
    
    
    words = set()
    documents = {}
    documenttok = {}
    
    for doc in os.listdir("./Abstracts/Abstracts"):
        path = os.path.join("./Abstracts/Abstracts", doc)
        no = int(doc.split('.')[0])
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        text = text.lower()
        temp = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(i) for i in temp if i.isalnum() and i not in stopwords]

        documents[no] = text
        documenttok[no] = tokens
        words.update(tokens)

    tf = {}
    df = defaultdict(int)
    idf = {}
    tfidf = {}
    documentvector = {}

    for i, tokens in documenttok.items():
        c = defaultdict(int)
        for t in tokens:
            c[t] += 1
        tf[i] = c
        for t in set(tokens):
            df[t] += 1

    totaldoc = len(documents)
    for t in words:
        idf[t] = math.log(totaldoc / df[t])

    for i, c in tf.items():
        tfidf[i] = {}
        for t, freq in c.items():
            tfidf[i][t] = freq * idf[t]

        temp2 = np.zeros(len(words))
        index = {t: d for d, t in enumerate(sorted(words))}
        for t, weight in tfidf[i].items():
            temp2[index[t]] = weight

        norm = np.linalg.norm(temp2)
        if norm > 0:
            temp2 = temp2 / norm

        documentvector[i] = temp2

    with open("index.pkl","wb") as f:
        pickle.dump({'documents': documents,'documenttok': documenttok,'words': words, 'tf': tf,'df': df,'idf': idf,'tfidf': tfidf,'documentvector': documentvector
        }, f)

    
    

# asking queries and providing the results 
# we will first preprocess the query
    
def results(query, alpha=0.05):
    
    
    query = query.lower()
    temp3 = word_tokenize(query)
    tokens =[]
    for t in temp3 :
        if t.isalnum() and t not in stopwords:
            tokens.append(t)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]   
    
    terms = {w:i for i, w in enumerate(sorted(words))}
    queryv = np.zeros(len(words))
    
    qtfreq = defaultdict(int)
    
    for t in tokens:
        if t in words: 
            qtfreq[t] += 1
    
    for qt,tf in qtfreq.items():
        if qt in idf:  
            index = terms[qt]
            queryv[index] =tf*idf[qt]
    
    norm = np.linalg.norm(queryv)
    if norm > 0:
        queryv = queryv/norm
    similarities = {}
    for i, docv in documentvector.items():
        similarity = np.dot(queryv,docv)
        if similarity >= alpha:
            similarities[i] = similarity
    sresult = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    return sresult
    
    

    
class ModernApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mini-Search-Engine")
        self.root.geometry("800x500")
        self.root.configure(bg="#22263d")
        self.root.resizable(True, True)
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#22263d")
        self.style.configure("TButton",background="#f5d273",foreground="#22263d",padding=10, borderwidth=0)
        self.style.map("TButton",background=[("active", "#0069d9"), ("pressed", "#0062cc")],foreground=[("active", "white"), ("pressed", "white")])
        self.style.configure("TLabel",background="#22263d")
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=False, padx=20, pady=20)
        self.header_label = ttk.Label(self.main_frame,text="Query system", font=("Segoe UI", 18, "bold",),foreground="#9194a8")
        self.header_label.pack(pady=(0, 20), anchor=tk.W)
        self.input_label = ttk.Label(self.main_frame, text="Enter query",foreground="#9194a8",font=("Segoe UI", 15, "bold",))
        self.input_label.pack(anchor=tk.W, pady=(0, 5))
        self.input_var = tk.StringVar()
        self.input_field = tk.Entry(self.main_frame,font=("Segoe UI", 11),bg="#9194a8",textvariable=self.input_var)
        self.input_field.pack(fill=tk.X, pady=(0, 15), ipady=8)
        self.button = ttk.Button(self.main_frame, text="Submit",command=self.on_submit)
        self.button.pack(anchor=tk.W, pady=(0, 20))
        self.output_frame = ttk.Frame(self.main_frame)
        self.output_frame.pack(fill=tk.BOTH, expand=True)
        self.output_label = ttk.Label(self.output_frame, text="Output:",foreground="#9194a8",font=("Segoe UI", 15, "bold",))
        self.output_label.pack(anchor=tk.W, pady=(0, 5))
        self.output_text = tk.Text(self.output_frame, height=8,font=("Segoe UI", 11),bg="#9194a8",padx=10,pady=10)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.config(state=tk.DISABLED)
        
    def on_submit(self):
        query = self.input_var.get()
    
        ans = results(query)
        answer = []
        j = min(40,len(ans))
        if j > 0:
            for i, (id, similarity) in enumerate(ans[:j]):
                answer.append(f"{i+1} {id} : score {similarity:.4f}")
        else:
            answer.append('nothing found')
        display = ''       
        for a in answer:
            display = display + '\n' + str(a)      
        
        if query:     
            
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Answer: \n{display}")
            self.output_text.config(state=tk.DISABLED)
            self.input_var.set("")  
        

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernApp(root)
    root.mainloop()