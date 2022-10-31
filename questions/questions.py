import nltk
import sys
import string
import os
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # create an empty dictionary to store everything
    files = dict()
    # list all of the files inside the corpus
    for docs in os.listdir(directory):
        # define path as
        path = os.path.join(directory, docs)
        # open the file
        with open(path, 'r', encoding="utf8") as f:
            # read the content
            content = f.read()
            # save the content in the dictionary
            files[docs] = content
    # return the dict
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    return [word for word in nltk.tokenize.word_tokenize(document.lower()) 
            if word not in nltk.corpus.stopwords.words("english") and word not in string.punctuation]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idf = dict()
    total_doc = len(documents)
    # find all the unique words in the corpus
    all_words = set(sum(documents.values(), []))
    # for each unique word in the corpus
    for word in all_words:
        # find the number of documents that contain that word
        # check each document to see if it has one
        doc_count = sum([1 for doc in documents if word in documents[doc]])
        # compute the idf value
        idf[word] = math.log(total_doc / doc_count)
    return idf


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    points = dict()
    for doc in files:
        points[doc] = 0
        for word in query:
            if word in files[doc]:
                points[doc] += idfs[word] * files[doc].count(word)
        if points[doc] == 0:
            del points[doc]
    # print(points)
    # print(sorted(points, key=points.get, reverse=True)[:n])
    return sorted(points, key=points.get, reverse=True)[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    points = dict()
    for sentence in sentences:
        points[sentence] = 0
        for word in query:
            if word in sentences[sentence]:
                points[sentence] += idfs[word]
        if len(sentence) > 0:
            if points[sentence] == 0:
                del points[sentence]
            else:
                density = len([word for word in query if word in sentences[sentence]]) / len(sentence)
                points[sentence] = (points[sentence], density)
    # print(points)
    # for sent in sorted(points, key=points.get, reverse=True)[:5]:
    #     print(sent, points[sent])

    return sorted(points, key=points.get, reverse=True)[:n] 


if __name__ == "__main__":
    main()
