from tkinter import *
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

global filename
global df

def upload():
    global filename, df
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Fill missing values with mode for each column
    df.fillna(df.mode().iloc[0], inplace=True)
    
    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded\n')
    text.insert(END, "Dataset Size: " + str(len(df)) + "\n")

def Distribution_of_rating(): 
    # Set the style and color palette
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Plot distribution of ratings
    sns.histplot(data=df, x='rating', bins=10, kde=True, color='skyblue')
    plt.title('Distribution of Ratings', fontsize=16)
    plt.xlabel('Rating', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.show()

def top10():
    global top_drugs
    top_drugs = df['drugName'].value_counts().head(10)
    
    # Set the style and color palette
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Plot top 10 drugs by review count
    sns.barplot(x=top_drugs.values, y=top_drugs.index, palette='viridis')
    plt.xlabel('Review Count', fontsize=14)
    plt.ylabel('Drug Name', fontsize=14)
    plt.title('Top 10 Drugs by Review Count', fontsize=16)
    plt.show()

def Frequent_conditions():
    global top_conditions
    top_conditions = df['condition'].value_counts().head(10)
    
    # Set the style and color palette
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Plot top 10 frequent conditions
    sns.barplot(x=top_conditions.values, y=top_conditions.index, palette='coolwarm')
    plt.xlabel('Frequency', fontsize=14)
    plt.ylabel('Condition', fontsize=14)
    plt.title('Top 10 Frequent Conditions', fontsize=16)
    plt.show()

def Model_Training():
    def submit_condition():
        user_condition = condition_entry.get()
        if user_condition:
            # Prepare the DataFrame
            df = pd.read_csv(filename)
            df = df[['drugName', 'condition']]
            df.dropna(subset=['condition'], inplace=True)

            # Create TF-IDF matrix
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(df['condition'])

            # Transform user input to TF-IDF vector
            user_condition_tfidf = tfidf_vectorizer.transform([user_condition])

            # Calculate cosine similarity
            similarity_scores = cosine_similarity(user_condition_tfidf, tfidf_matrix)

            # Get top recommended medicines
            top_indices = similarity_scores.argsort()[0][::-1]
            top_medicines = df['drugName'].iloc[top_indices]

            # Display top recommended medicines in the main text box
            text.delete('1.0', END)
            text.insert(END, f"Top recommended medicines for {user_condition}:\n")
            for medicine in top_medicines:
                text.insert(END, f"{medicine}\n")

    # Create a new window for input
    input_window = Toplevel(main)
    input_window.title("Enter Health Condition")
    input_window.geometry("400x200")
    input_window.configure(bg='#e0f7fa')

    # Create input label and entry widget
    condition_label = Label(input_window, text="Enter your health condition:", bg='#e0f7fa', font=('Helvetica', 12, 'bold'))
    condition_label.pack(pady=10)
    condition_entry = Entry(input_window, width=40, font=('Helvetica', 12))
    condition_entry.pack(pady=10)

    # Create submit button
    submit_button = Button(input_window, text="Submit", command=submit_condition, bg='#26c6da', fg='white', font=('Helvetica', 12, 'bold'))
    submit_button.pack(pady=20)

def top10req():
    def submit_condition():
        user_condition = condition_entry.get()
        if user_condition:
            # Prepare the DataFrame
            df = pd.read_csv(filename)
            df = df[['drugName', 'condition']]
            df.dropna(subset=['condition'], inplace=True)

            # Create TF-IDF matrix
            tfidf_vectorizer = TfidfVectorizer()
            tfidf_matrix = tfidf_vectorizer.fit_transform(df['condition'])

            # Transform user input to TF-IDF vector
            user_condition_tfidf = tfidf_vectorizer.transform([user_condition])

            # Calculate cosine similarity
            similarity_scores = cosine_similarity(user_condition_tfidf, tfidf_matrix)

            # Get top 10 recommended medicines
            top_indices = similarity_scores.argsort()[0][::-1][:10]
            top_medicines = df['drugName'].iloc[top_indices]

            # Display top recommended medicines in the main text box
            text.delete('1.0', END)
            text.insert(END, f"Top 10 recommended medicines for {user_condition}:\n")
            for medicine in top_medicines:
                text.insert(END, f"{medicine}\n")

    # Create a new window for input
    input_window = Toplevel(main)
    input_window.title("Enter Health Condition")
    input_window.geometry("400x200")
    input_window.configure(bg='#e0f7fa')

    # Create input label and entry widget
    condition_label = Label(input_window, text="Enter your health condition:", bg='#e0f7fa', font=('Helvetica', 12, 'bold'))
    condition_label.pack(pady=10)
    condition_entry = Entry(input_window, width=40, font=('Helvetica', 12))
    condition_entry.pack(pady=10)

    # Create submit button
    submit_button = Button(input_window, text="Submit", command=submit_condition, bg='#26c6da', fg='white', font=('Helvetica', 12, 'bold'))
    submit_button.pack(pady=20)

# Main window
main = tk.Tk()
main.title("Drug Recommendation System based on Sentiment Analysis of Drug Reviews using Machine Learning") 
main.geometry("1600x1500")
main.configure(bg='#32d1a7')

font = ('times', 16, 'bold')
title = tk.Label(main, text='Drug Recommendation System based on Sentiment Analysis of Drug Reviews using Machine Learning', font=("times"))
title.config(bg='Dark Blue', fg='white')
title.config(font=font)           
title.config(height=3, width=145)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = tk.Text(main, height=20, width=180)
scroll = tk.Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

uploadButton = tk.Button(main, text="Upload Dataset", command=upload, bg="sky blue", width=15)
uploadButton.place(x=50, y=600)
uploadButton.config(font=font1)

pathlabel = tk.Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)     
pathlabel.place(x=250, y=600)

Distribution_of_rating = tk.Button(main, text="Distribution of Ratings", command=Distribution_of_rating, bg="light green", width=20)
Distribution_of_rating.place(x=50, y=650)
Distribution_of_rating.config(font=font1)


Frequent_conditions = tk.Button(main, text="Frequent Conditions", command=Frequent_conditions, bg="pink", width=20)
Frequent_conditions.place(x=250, y=650)  # Adjusted y-coordinate
Frequent_conditions.config(font=font1)


top10_drogs = tk.Button(main, text="Top 10 Drugs", command=top10, bg="lightgrey", width=15)
top10_drogs.place(x=450, y=650)  
top10_drogs.config(font=font1)


rec_1 = tk.Button(main, text="Model Training", command=Model_Training, bg="yellow", width=15)
rec_1.place(x=630, y=650)  
rec_1.config(font=font1)

# Button for top 10 drug recommendation
top10req_button = tk.Button(main, text="Top 10 Drug Recommendation", command=top10req, bg="lightgreen", width=25)
top10req_button.place(x=830, y=650)
top10req_button.config(font=font1)

main.config(bg='#32d1a7')
main.mainloop()
