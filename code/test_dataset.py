import torch
from mvfoul_dataset import MVFoulDataset

def main():
    # root = "/path/to/SoccerNetData/mvfouls"
    root = "/Users/zubin/Desktop/CS566/SoccerNetData/mvfouls"

    dataset = MVFoulDataset(
        root=root,
        split="train",
        num_frames=16,
    )

    print("Dataset size:", len(dataset))

    clip, sev, ftype = dataset[0]
    print("Clip shape:", clip.shape)          # expect (T, C, H, W)
    print("Severity label:", sev)             # int in [0..3]
    print("Foul type label:", ftype)          # int (0..10)

    # Quick check for a random index
    idx = torch.randint(low=0, high=len(dataset), size=(1,)).item()
    clip2, sev2, ftype2 = dataset[idx]
    print(f"\nRandom sample idx {idx}:")
    print("  Clip shape:", clip2.shape)
    print("  Severity:", sev2)
    print("  Foul type:", ftype2)

if __name__ == "__main__":
    main()