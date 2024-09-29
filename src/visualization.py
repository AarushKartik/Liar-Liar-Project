def plot_word_cloud(text, title=None):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

# Create a combined text for Word Cloud from training data
all_text = ' '.join(X_train)
plot_word_cloud(all_text, title='Word Cloud for Training Data')

# Class Distribution
def plot_class_distribution(y):
    class_labels, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar(class_labels, counts, color='skyblue')
    plt.xticks(class_labels)
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.show()

# Plot class distribution for training labels
plot_class_distribution(y_train_array)
