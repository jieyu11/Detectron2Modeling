import json
from pathlib import Path

class CocoFilter():
    """ Filters the COCO dataset
    """
    def main(self, args):
        # Open json
        self.input_json_path = Path(args.input_json)
        # Verify input path exists
        if not self.input_json_path.exists():
            print('Input json path not found.')
            print('Quitting early.')
            quit()
        # Load the json
        print('Loading json file...')
        with open(self.input_json_path) as json_file:
            self.coco = json.load(json_file)
        print('info', self.coco["info"])
        print('licenses', self.coco["licenses"])
        print('images', self.coco['images'][0])
        print('annotations', self.coco['annotations'][0])
        print('categories', self.coco["categories"])
        print("N images", len(self.coco['images']))
        print("N annotations", len(self.coco['annotations']))

        instance_counts = {idx["id"]: 0 for idx in self.coco["categories"]}
        for anndict in self.coco["annotations"]:
            cidx = anndict["category_id"]
            instance_counts[cidx] += 1
        print("Instance counts:", instance_counts)

if __name__ == "__main__":
    import argparse

    # to run:
    # python filter_balance_instance.py -i person_sportsball_val2014.json \
    # -o person_sportsball_val2014_balanced.json
    # 
    parser = argparse.ArgumentParser(description="Filter COCO JSON: "
    "Filters a COCO Instances JSON file to only include specified categories. "
    "This includes images, and annotations. Does not modify 'info' or 'licenses'.")
    
    parser.add_argument("-i", "--input_json", dest="input_json",
        help="path to a json file in coco format")
    args = parser.parse_args()

    cf = CocoFilter()
    cf.main(args)
