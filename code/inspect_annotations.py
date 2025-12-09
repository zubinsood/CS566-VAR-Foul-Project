import os
import json

def inspect_annotations(root, split="train"):
    ann_path = os.path.join(root, split, "annotations.json")
    print("Loading:", ann_path)
    with open(ann_path, "r") as f:
        data = json.load(f)

    print("Type of top-level object:", type(data))

    if not isinstance(data, dict):
        print("Expected dict at top level, got:", type(data))
        return

    print("Top-level keys:", list(data.keys()))

    if "Actions" not in data:
        print("No 'Actions' key found in annotations.json")
        return

    actions = data["Actions"]
    print("Type of 'Actions':", type(actions))

    # Look through keys in the dict structure
    if isinstance(actions, dict):
        action_keys = list(actions.keys())
        print("Number of actions listed:", len(action_keys))
        print("First 5 action keys:", action_keys[:5])

        first_key = action_keys[0]
        first_entry = actions[first_key]

        print("\nFirst action key:", first_key)
        print("First action entry (raw):")
        print(first_entry)

        print("\nFirst action entry keys and values:")
        for k, v in first_entry.items():
            print(f"  {k}: {v}")
    else:
        print("Unexpected type for 'Actions':", type(actions))

if __name__ == "__main__":
    # root="/path/to/SoccerNetData/mvfouls"
    root = "/Users/zubin/Desktop/CS566/SoccerNetData/mvfouls"
    inspect_annotations(root, split="train")