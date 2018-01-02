from lemmatizing import lemmatize
from clustering import cluster, Data
import os
from numpy import argsort


def main():
    identifiers = []
    dreams = []
    ldreams = []
    with open('dreams.txt', encoding='ascii', errors = 'ignore') as dreamstxt:
        for i, line in enumerate(dreamstxt):
            if i % 3 == 0:
                identifiers.append(line[:-1])
            elif (i - 1) % 3 == 0:
                dreams.append(line)
    lemmatize(identifiers, dreams)
    with open('lemmatized.txt', encoding='ascii') as lemmatized:
        for i, line in enumerate(lemmatized):
            if (i - 1) % 3 == 0:
                ldreams.append(line)
    data = cluster(identifiers, ldreams)

    result_path = os.path.join(os.getcwd(),'clusters')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    else: #clear any previously stored files
        for root, dirs, files in os.walk(result_path, topdown = False):
            for name in files:
                os.remove(os.path.join(root, name))
    os.chdir(result_path)

    for label in range(data.cluster_count):
        
        defining_features = []          
        for i, f_count in enumerate(data.feature_counts[label]):
            if f_count >= data.min_samples - 1:
                defining_features.append(i)
        if len(defining_features) > 20:
            defining_features = argsort(data.feature_counts[label])[-20:][::-1]
        
        common_words = list(map(lambda x: data.vocab[x], defining_features))
        
        samples = list(map(lambda x: identifiers[x], data.sample_indeces[label]))
        
        print(label)
        print("Common features:\n", common_words)
        print("Dreams:\n", samples)

        with open(str(label) + '.txt', 'w') as results:
            results.write("Common features:\n")
            for word in common_words:
                results.write(word + "\n")
            results.write("\nDreams:\n")
            sample_dreams = list(map(lambda x: dreams[x], data.sample_indeces[label]))
            for i in range(len(samples)):
                results.write(samples[i] + "\n" + sample_dreams[i] + "\n\n")


if __name__ == "__main__":
    main()
