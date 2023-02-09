import finetuner
finetuner.login()

from docarray import Document, DocumentArray

da = DocumentArray.from_files('../color/*/*.*')

print(da)
# for i in da:
#     print(i.uri.split('/')[8])

def assign_labels(d: Document):
    d.tags['finetuner_label'] = d.uri.split('/')[2]
    return d

da.apply(assign_labels, show_progress=True)
# shuffle and train-test-split to 50-50
da = da.shuffle()
train_da = da

test_da = train_da



####

import hubble

client = hubble.Client()

try:
    client.get_user_info()
except hubble.excepts.AuthenticationRequiredError:
    print('Please login first.')
except Exception:
    print('Unknown error')

print(client.get_user_info())


pairs = DocumentArray() # initialize a DocumentArray as final training data.

prompt = 'This is a photo of '
for doc in train_da:
    pair = Document()
    img_chunk = doc.load_uri_to_image_tensor(224, 224)
    img_chunk.modality = 'image'
    txt_chunk = Document(content=prompt + doc.tags['finetuner_label'])
    txt_chunk.modality = 'text'
    pair.chunks.extend([img_chunk, txt_chunk])
    # add pair to pairs
    pairs.append(pair)



run = finetuner.fit(
    model='openai/clip-vit-base-patch32', # fine-tune CLIP
    train_data=pairs,
    learning_rate=1e-5,
    loss='CLIPLoss'
)

for entry in run.stream_logs():
    print(entry)


artifact = run.save_artifact('./finetuned_model/')

