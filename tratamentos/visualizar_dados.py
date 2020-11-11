def plotaFrequenciaDePalavras(data, column, n, flagImprimeInformacoesTela):
    from sklearn.feature_extraction.text import CountVectorizer
    import collections
    from wordcloud import WordCloud
    
    cv = CountVectorizer()
    #column = input("Select the column:\n")
    bow = cv.fit_transform(data[column].values)
    word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
    word_counter = collections.Counter(word_freq)
    word_counter_df = pd.DataFrame(word_counter.most_common(n), columns = ['word', 'freq'])
    fig, ax = plt.subplots(figsize=(12, 10))
    g= sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    plt.show();
    
    if (flagImprimeInformacoesTela == True):
        word_counter.most_common(10)
    
    #Visualização das palavras depois do processo de limpeza 
    #column = input("Select a column to the visualization:\n")
    wc_string = data[column].str.cat(sep=' ')
    
    wc = WordCloud(width=1600, height=800,background_color="white", max_words=2000).generate(wc_string)
    plt.figure(figsize=(10,10))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title('Cloud Words', fontsize=18)
    plt.show()