import json
from pathlib import Path

class CocoFilter():
    """ Filters the COCO dataset
    """
    def main(self, args):
        # Open json
        self.input_json_path = Path(args.input_json)
        self.output_json_path = Path(args.output_json)

        # Verify input path exists
        if not self.input_json_path.exists():
            print('Input json path not found.')
            print('Quitting early.')
            quit()
        # Load the json
        print('Loading json file...')
        with open(self.input_json_path) as json_file:
            self.coco = json.load(json_file)
        
        print("Totoal number of images: %d" % len(self.coco["images"]))
        print("Categories", self.coco["categories"])
        # select the images that has category index of 
        image_ids_keep = set()
        for anndict in self.coco["annotations"]:
            if anndict["category_id"] == args.priority_category:
                image_ids_keep.add(anndict["image_id"])
        print("Number of images to be saved: %d" % len(image_ids_keep))

        self.new_images = []
        for imgdict in self.coco["images"]:
            if imgdict["id"] in image_ids_keep:
                self.new_images.append(imgdict)

        self.new_segmentations = []
        instance_counts = {idx["id"]: 0 for idx in self.coco["categories"]}
        for anndict in self.coco["annotations"]:
            if anndict["image_id"] not in image_ids_keep:
                continue
            cidx = anndict["category_id"]
            instance_counts[cidx] += 1
            self.new_segmentations.append(anndict)
        print("Instance counts:", instance_counts)
        # Build new JSON
        new_master_json = {
            'info': self.coco["info"],
            'licenses': self.coco["licenses"],
            'images': self.new_images,
            'annotations': self.new_segmentations,
            'categories': self.coco["categories"]
        }

        # Write the JSON to a file
        print('Saving new json file...')
        with open(self.output_json_path, 'w+') as output_file:
            json.dump(new_master_json, output_file)

        print('Filtered json saved.')

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
    parser.add_argument("-o", "--output_json", dest="output_json",
        help="path to save the output json")
    parser.add_argument("-p", "--priority-category", dest="priority_category",
        default=2, type=int, required=False,
        help="Select images in particular with priority-category")
    args = parser.parse_args()

    cf = CocoFilter()
    cf.main(args)
