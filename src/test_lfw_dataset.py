from sklearn.datasets import fetch_lfw_people

print("Loading LFW dataset...")

lfw = fetch_lfw_people(
    min_faces_per_person=2,
    resize=0.5,
    color=True
)

print("Dataset loaded successfully!")
print("Number of images:", len(lfw.images))
print("Number of identities:", len(lfw.target_names))
print("Image shape:", lfw.images.shape)
print("First 10 identity names:", lfw.target_names[:10])