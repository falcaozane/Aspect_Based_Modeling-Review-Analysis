import streamlit as st
import pandas as pd
import spacy
import re
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import operator
import matplotlib.pyplot as plt
from wordcloud import WordCloud



# Load English language model
nlp = spacy.load("en_core_web_lg")

# List of CSV files
csv_files = ['listerine.csv', 'mouthwash.csv', 'toothbrush.csv', 'toothpaste.csv']

# Function to generate word cloud
def generate_wordcloud(aspect_keywords):
    # Concatenate all aspect keywords into a single string
    text = ' '.join(aspect_keywords)

    # Create WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Plot the word cloud
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Aspect Keywords Word Cloud')
    st.pyplot(fig)

# Define the aspects and keywords list
aspects_keywords_list = [
    "Whitening", "Plaque removal", "Cavity prevention", "Freshness/breath", "Tartar control", "Sensitivity relief",
    "Convenience", "Application", "Packaging", "Accessibility", "Instructions",
    "Minty", "Sweet", "Refreshing", "Bitter", "Strong", "Artificial",
    "Gentle on gums", "Non-irritating", "Soft bristles/texture", "Smooth application",
    "Affordable", "Cost-effective", "Expensive", "Worth the price", "Budget-friendly",
    "Lasting freshness", "Long-lasting effects", "Durable packaging", "Staying power",
    "Natural ingredients", "Chemical-free", "Artificial additives", "Alcohol-free", "Fluoride-free",
    "Creamy", "Foamy", "Gel-like", "Thick", "Thin", "Grainy",
    "Fresh", "Strong scent", "Overpowering", "Unpleasant odor",
    "Design", "Size", "Portability", "Environmental friendliness"
]


def process_reviews(data):
    aspect_terms = []
    comp_terms = []
    easpect_terms = []
    ecomp_terms = []
    enemy = []
    
    for x in tqdm(range(len(data['Review Text']))):
        amod_pairs = []
        advmod_pairs = []
        compound_pairs = []
        xcomp_pairs = []
        neg_pairs = []
        eamod_pairs = []
        eadvmod_pairs = []
        ecompound_pairs = []
        eneg_pairs = []
        excomp_pairs = []
        enemlist = []
        if len(str(data['Review Text'][x])) != 0:
            lines = str(data['Review Text'][x]).replace('*',' ').replace('-',' ').replace('so ',' ').replace('be ',' ').replace('are ',' ').replace('just ',' ').replace('get ','').replace('were ',' ').replace('When ','').replace('when ','').replace('again ',' ').replace('where ','').replace('how ',' ').replace('has ',' ').replace('Here ',' ').replace('here ',' ').replace('now ',' ').replace('see ',' ').replace('why ',' ').split('.')
            for line in lines:
                enem_list = []
                for eny in aspects_keywords_list:
                    enem = re.search(eny,line)
                    if enem is not None:
                        enem_list.append(enem.group())
                if len(enem_list)==0:
                    doc = nlp(line)
                    str1=''
                    str2=''
                    for token in doc:
                        if token.pos_ == 'NOUN':
                            for j in token.lefts:
                                if j.dep_ == 'compound':
                                    compound_pairs.append((j.text+' '+token.text,token.text))
                                if j.dep_ == 'amod' and j.pos_ == 'ADJ': #primary condition
                                    str1 = j.text+' '+token.text
                                    amod_pairs.append(j.text+' '+token.text)
                                    for k in j.lefts:
                                        if k.dep_ == 'advmod': #secondary condition to get adjective of adjectives
                                            str2 = k.text+' '+j.text+' '+token.text
                                            amod_pairs.append(k.text+' '+j.text+' '+token.text)
                                    mtch = re.search(re.escape(str1),re.escape(str2))
                                    if mtch is not None:
                                        amod_pairs.remove(str1)
                        if token.pos_ == 'VERB':
                            for j in token.lefts:
                                if j.dep_ == 'advmod' and j.pos_ == 'ADV':
                                    advmod_pairs.append(j.text+' '+token.text)
                                if j.dep_ == 'neg' and j.pos_ == 'ADV':
                                    neg_pairs.append(j.text+' '+token.text)
                            for j in token.rights:
                                if j.dep_ == 'advmod'and j.pos_ == 'ADV':
                                    advmod_pairs.append(token.text+' '+j.text)
                        if token.pos_ == 'ADJ':
                            for j,h in zip(token.rights,token.lefts):
                                if j.dep_ == 'xcomp' and h.dep_ != 'neg':
                                    for k in j.lefts:
                                        if k.dep_ == 'aux':
                                            xcomp_pairs.append(token.text+' '+k.text+' '+j.text)
                                elif j.dep_ == 'xcomp' and h.dep_ == 'neg':
                                    if k.dep_ == 'aux':
                                            neg_pairs.append(h.text +' '+token.text+' '+k.text+' '+j.text)

                else:
                    enemlist.append(enem_list)
                    doc = nlp(line)
                    str1=''
                    str2=''
                    for token in doc:
                        if token.pos_ == 'NOUN':
                            for j in token.lefts:
                                if j.dep_ == 'compound':
                                    ecompound_pairs.append((j.text+' '+token.text,token.text))
                                if j.dep_ == 'amod' and j.pos_ == 'ADJ': #primary condition
                                    str1 = j.text+' '+token.text
                                    eamod_pairs.append(j.text+' '+token.text)
                                    for k in j.lefts:
                                        if k.dep_ == 'advmod': #secondary condition to get adjective of adjectives
                                            str2 = k.text+' '+j.text+' '+token.text
                                            eamod_pairs.append(k.text+' '+j.text+' '+token.text)
                                    mtch = re.search(re.escape(str1),re.escape(str2))
                                    if mtch is not None:
                                        eamod_pairs.remove(str1)
                        if token.pos_ == 'VERB':
                            for j in token.lefts:
                                if j.dep_ == 'advmod' and j.pos_ == 'ADV':
                                    eadvmod_pairs.append(j.text+' '+token.text)
                                if j.dep_ == 'neg' and j.pos_ == 'ADV':
                                    eneg_pairs.append(j.text+' '+token.text)
                            for j in token.rights:
                                if j.dep_ == 'advmod'and j.pos_ == 'ADV':
                                    eadvmod_pairs.append(token.text+' '+j.text)
                        if token.pos_ == 'ADJ':
                            for j in token.rights:
                                if j.dep_ == 'xcomp':
                                    for k in j.lefts:
                                        if k.dep_ == 'aux':
                                            excomp_pairs.append(token.text+' '+k.text+' '+j.text)
            pairs = list(set(amod_pairs+advmod_pairs+neg_pairs+xcomp_pairs))
            epairs = list(set(eamod_pairs+eadvmod_pairs+eneg_pairs+excomp_pairs))
            for i in range(len(pairs)):
                if len(compound_pairs)!=0:
                    for comp in compound_pairs:
                        mtch = re.search(re.escape(comp[1]),re.escape(pairs[i]))
                        if mtch is not None:
                            pairs[i] = pairs[i].replace(mtch.group(),comp[0])
            for i in range(len(epairs)):
                if len(ecompound_pairs)!=0:
                    for comp in ecompound_pairs:
                        mtch = re.search(re.escape(comp[1]),re.escape(epairs[i]))
                        if mtch is not None:
                            epairs[i] = epairs[i].replace(mtch.group(),comp[0])

        aspect_terms.append(pairs)
        comp_terms.append(compound_pairs)
        easpect_terms.append(epairs)
        ecomp_terms.append(ecompound_pairs)
        enemy.append(enemlist)
    
    return aspect_terms, comp_terms, easpect_terms, ecomp_terms, enemy

def analyze_sentiment(data):
    sentiment = []
    analyser = SentimentIntensityAnalyzer()
    
    for i in range(len(data)):
        score_dict = {'pos': 0, 'neg': 0, 'neu': 0}
        if len(data['aspect_keywords'][i]) != 0:
            for aspect in data['aspect_keywords'][i]:
                sent = analyser.polarity_scores(aspect)
                score_dict['neg'] += sent['neg']
                score_dict['pos'] += sent['pos']
                score_dict['neu'] += sent['neu']
            
            sentiment.append(max(score_dict.items(), key=operator.itemgetter(1))[0])
        else:
            sentiment.append('neu')
    
    return sentiment

def main():
    st.title("Aspect_Based_Modeling Review Analysis")
    
    # Sidebar for selecting CSV file
    selected_csv = st.sidebar.selectbox("Select CSV file", csv_files)
    
    
    
    # Read selected CSV file
    if selected_csv:
        data = pd.read_csv(selected_csv)
        
        # Preprocess reviews
        aspect_terms, comp_terms, easpect_terms, ecomp_terms, enemy = process_reviews(data)
        data['aspect_keywords'] = aspect_terms
        
        # Sentiment Analysis
        sentiment = analyze_sentiment(data)
        data['sentiment'] = sentiment
        
        # Display the dataframe
        st.write(data)

        st.subheader("Aspect Words")
        # Display the dataframe and aspect words side by side
        col1, col2, col3 = st.columns([3, 2, 1])
        
        
        
        for aspect_list, sentiment_label in zip(aspect_terms, sentiment):
            for aspect_word in aspect_list:
                if sentiment_label == 'pos':
                    with col1:
                        st.markdown(f'<span style="color:green">{aspect_word}</span>', unsafe_allow_html=True)
                elif sentiment_label == 'neg':
                    with col2:
                        st.markdown(f'<span style="color:red">{aspect_word}</span>', unsafe_allow_html=True)
                else:
                    with col3:
                        st.markdown(f'<span style="color:yellow">{aspect_word}</span>', unsafe_allow_html=True)

        st.subheader("Sentiment Distribution")
        # Bar graph for sentiment distribution
        sentiment_counts = data['sentiment'].value_counts()

        fig, ax = plt.subplots()
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'yellow'])
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        ax.set_title('Sentiment Distribution')

        st.pyplot(fig)

        # Word Cloud
        generate_wordcloud(aspects_keywords_list)

    

    
    

if __name__ == "__main__":
    main()
