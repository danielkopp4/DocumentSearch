import sys
import os
import pickle
from typing import List 
from lib.webscraping import NYLawWebscraping
from lib.semantic_search import Querier, model
from lib.data_representations import NYLaw, add_law, Node
from progressbar import Percentage, Bar, ETA, ProgressBar



class Settings:
    base_url = "https://www.nysenate.gov{}"
    origin_site = "/legislation/laws/CONSOLIDATED"
    save_path = "./data/big/ny_laws"
    sites_save_file = "consolidated_urls.txt"
    parsed_save_file = "parsed.P"
    search_tree_loc = "search"
    search_pickle = ".P"

def make_folders(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_sites_filename() -> str:
    make_folders(Settings.save_path)
    return os.path.join(Settings.save_path, Settings.sites_save_file)


def get_parsed_data_filename() -> str:
    make_folders(Settings.save_path)
    return os.path.join(Settings.save_path, Settings.parsed_save_file)


def get_search_tree_file_path(pickle: bool = False) -> str:
    make_folders(Settings.save_path)
    partial_path = os.path.join(Settings.save_path, Settings.search_tree_loc)
    if pickle:
        return partial_path + Settings.search_pickle
    return partial_path



def write_to_sites(sites: List[str]) -> None:
    full_path = get_sites_filename()
    with open(full_path, "w") as file:
        file.write("\n".join(sites))
        file.write("\n")
    

def read_from_sites() -> List[str]:
    full_path = get_sites_filename()
    sites = []
    with open(full_path, "r") as file:
        sites = file.readlines()
    return sites



def save_parsed_data(dataset: Node) -> None:
    full_path = get_parsed_data_filename()
    with open(full_path, "wb") as file:
        pickle.dump(dataset, file)
    

def load_parsed_data() -> Node:
    full_path = get_parsed_data_filename()
    dataset = None
    with open(full_path, "rb") as file:
        dataset = pickle.load(file)
    return dataset



def save_search_tree(querier: Querier) -> None:
    full_path = get_search_tree_file_path(True)
    with open(full_path, "wb") as file:
        pickle.dump(querier, file)


def load_search_tree() -> Querier:
    full_path = get_search_tree_file_path(True)
    querier = None
    with open(full_path, "rb") as file:
        querier = pickle.load(file)
    return querier


def load_laws() -> None:
    sites = NYLawWebscraping.get_all_end_sites(Settings.base_url, Settings.origin_site)
    write_to_sites(sites)

    print("completed...")
    print("wrote to '{}'".format(get_sites_filename()))


def parse_laws() -> None:
    sites = read_from_sites()
    dataset = NYLaw("laws")

    widgets = ['Loading Data: ', Percentage(), ' ', Bar(), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=len(sites)).start()
    
    for i, identifier in enumerate(sites):
        contents = NYLawWebscraping.get_law_contents(Settings.base_url, identifier)
        
        if contents != None:
            add_law(dataset, identifier, contents)

        pbar.update(i+1)

    pbar.finish()

    print("data parsing complete...")
    print("saving...")

    save_parsed_data(dataset)

    print("complete...")
    print("saved to '{}'".format(get_parsed_data_filename()))


def encode_and_build() -> None:
    dataset = load_parsed_data()
    path = get_search_tree_file_path()
    querier = Querier(model, path)

    print("building search tree...")

    querier.build(dataset.get_nodes())

    print("finished building")
    print("saving...")

    save_search_tree(querier)

    print("saved to '{}'".format(get_search_tree_file_path(True)))


def search() -> None:
    querier = load_search_tree()
    querier.restore_ann_enc()
    print("Loaded search tree with {} nodes".format(len(querier.nodes)))

    should_end = False
    while not should_end:
        inp = input("Enter search query: ")
        if inp == "end" or inp == "q" or inp == "quit":
            should_end = True
            continue

        print(querier.query(inp, top_k=1))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        search()
    else:
        if sys.argv[1] == "load":
            load_laws()
        elif sys.argv[1] == "parse":
            parse_laws()
        elif sys.argv[1] == "build":
            encode_and_build()
        elif sys.argv[1] == "search":
            search()
        else:
            print("'{}' is not a valid argument".format(sys.argv[1]))
            print("try: load, parse, build, search")