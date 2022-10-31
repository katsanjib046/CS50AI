import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = dict()
    links = corpus[page]
    # if the current page has no links
    if len(links) == 0:
        for page in corpus:
            distribution[page] = 1 / len(corpus)
    # if the current page has links
    else:
        for link in corpus:
            if link in links:
                distribution[link] = damping_factor / len(links)
                distribution[link] += (1 - damping_factor) / len(corpus)
            else:   
                distribution[link] = (1 - damping_factor) / len(corpus)
    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ranks = dict()
    first_sample = random.choice(list(corpus.keys()))
    ranks[first_sample] = 1
    for i in range(n-1):
        page = random.choices(list(ranks.keys()), list(ranks.values()))[0]
        distribution = transition_model(corpus, page, damping_factor)
        next_page = random.choices(list(distribution.keys()), list(distribution.values()))[0]
        if next_page in ranks:
            ranks[next_page] += 1
        else:
            ranks[next_page] = 1
        # normalize the ranks
    for page in ranks:
        ranks[page] /= n
    return ranks



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Create an empty dictionary for ranks
    ranks = dict()
    # initialize ranks with default value of 1/N
    for page in corpus:
        ranks[page] = 1 / len(corpus)
    # iterate until convergence (ranks don't change by more than 0.001)
    while True:
        new_ranks = dict()
        for page in corpus:
            new_ranks[page] = (1 - damping_factor) / len(corpus)
            for link in corpus:
                if page in corpus[link]:
                    new_ranks[page] += damping_factor * ranks[link] / len(corpus[link])
        # check for convergence
        if all(abs(new_ranks[page] - ranks[page]) <= 0.001 for page in corpus):
            return new_ranks
        else:
            ranks = new_ranks


if __name__ == "__main__":
    main()
